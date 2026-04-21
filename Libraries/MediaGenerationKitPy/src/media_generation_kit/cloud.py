from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .errors import MediaGenerationKitError

DEFAULT_CLOUD_BASE_URL = "https://api.drawthings.ai"
DEFAULT_CLOUD_COMPUTE_HOST = "compute.drawthings.ai"
DEFAULT_CLOUD_COMPUTE_PORT = 443
DEFAULT_HTTP_USER_AGENT = "MediaGenerationKitPy/0.1"
_PAYG_CACHE_TTL = 30.0


class CloudAuthenticator:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_CLOUD_BASE_URL,
        token_refresh_threshold: float = 300.0,
        request_timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = _normalize_base_url(base_url)
        self.token_refresh_threshold = token_refresh_threshold
        self.request_timeout = request_timeout

        self._cached_short_term_token: str | None = None
        self._token_expiry_monotonic: float | None = None
        self._cached_remote_models: set[str] = set()
        self._in_flight_token_request: asyncio.Task[tuple[str, int]] | None = None
        self._lock = asyncio.Lock()

    async def short_term_token(self) -> str:
        async with self._lock:
            if (
                self._cached_short_term_token is not None
                and self._token_expiry_monotonic is not None
                and self._token_expiry_monotonic - time.monotonic() > self.token_refresh_threshold
            ):
                return self._cached_short_term_token

            task = self._in_flight_token_request
            if task is None:
                task = asyncio.create_task(asyncio.to_thread(self._fetch_short_term_token))
                self._in_flight_token_request = task

        try:
            token, expires_in = await task
        except Exception as error:
            async with self._lock:
                if self._in_flight_token_request is task:
                    self._in_flight_token_request = None
            if isinstance(error, MediaGenerationKitError):
                raise
            raise MediaGenerationKitError.generation_failed(
                f"cloud token request failed: {error}"
            ) from error

        async with self._lock:
            self._cached_short_term_token = token
            self._token_expiry_monotonic = time.monotonic() + max(int(expires_in), 0)
            if self._in_flight_token_request is task:
                self._in_flight_token_request = None
        return token

    def update_remote_models_from_handshake(self, models: set[str] | list[str] | tuple[str, ...]) -> None:
        self._cached_remote_models = set(models)

    def remote_models(self) -> set[str]:
        return set(self._cached_remote_models)

    def _fetch_short_term_token(self) -> tuple[str, int]:
        body = _json_body(
            {
                "apiKey": self.api_key,
                "appCheckType": "none",
            }
        )
        request = urllib.request.Request(
            _join_url(self.base_url, "/sdk/token"),
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_HTTP_USER_AGENT,
            },
            method="POST",
        )
        payload = _read_json_response(request, timeout=self.request_timeout)
        token = payload.get("shortTermToken")
        expires_in = payload.get("expiresIn")
        if not isinstance(token, str) or not token:
            raise MediaGenerationKitError.generation_failed("cloud token response missing shortTermToken")
        if not isinstance(expires_in, int):
            raise MediaGenerationKitError.generation_failed("cloud token response missing expiresIn")
        return token, expires_in


class CloudAuthenticatorRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._authenticators: dict[tuple[str, str], CloudAuthenticator] = {}

    def authenticator(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_CLOUD_BASE_URL,
    ) -> CloudAuthenticator:
        normalized_base_url = _normalize_base_url(base_url)
        key = (api_key, normalized_base_url)
        with self._lock:
            authenticator = self._authenticators.get(key)
            if authenticator is None:
                authenticator = CloudAuthenticator(api_key, base_url=normalized_base_url)
                self._authenticators[key] = authenticator
            return authenticator


CloudAuthenticatorRegistry.shared = CloudAuthenticatorRegistry()  # type: ignore[attr-defined]

_payg_cache_lock = threading.Lock()
_payg_status_cache: dict[tuple[str, str], tuple[bool, float]] = {}


def resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    env_value = os.environ.get("DRAWTHINGS_API_KEY")
    if env_value:
        return env_value
    raise MediaGenerationKitError.generation_failed(
        "cloud compute requires an api_key or DRAWTHINGS_API_KEY"
    )


async def prefetch_payg_enabled(short_term_token: str, *, base_url: str = DEFAULT_CLOUD_BASE_URL) -> bool:
    cached = _cached_payg_enabled(short_term_token, base_url)
    if cached is not None:
        return cached
    return await asyncio.to_thread(
        _fetch_payg_enabled,
        short_term_token,
        _normalize_base_url(base_url),
        10.0,
    )


async def cloud_authenticate(
    *,
    short_term_token: str,
    encoded_blob: str,
    from_bridge: bool,
    estimated_compute_units: float | None,
    base_url: str = DEFAULT_CLOUD_BASE_URL,
    timeout: float = 30.0,
) -> str:
    normalized_base_url = _normalize_base_url(base_url)
    payg_enabled = _cached_payg_enabled(short_term_token, normalized_base_url)
    if payg_enabled is None:
        payg_enabled = await asyncio.to_thread(
            _fetch_payg_enabled,
            short_term_token,
            normalized_base_url,
            min(timeout, 10.0),
        )

    is_positive_amount = (estimated_compute_units or 0) > 0
    body = _json_body(
        {
            "blob": encoded_blob,
            "fromBridge": from_bridge,
            "attestationSupported": False,
            "assertionPayload": None,
            "originalTransactionId": None,
            "isSandbox": False,
            "consumableType": "payg" if payg_enabled and is_positive_amount else None,
            "amount": float(estimated_compute_units) if is_positive_amount else None,
        }
    )
    request = urllib.request.Request(
        _join_url(normalized_base_url, "/authenticate"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": short_term_token,
            "User-Agent": DEFAULT_HTTP_USER_AGENT,
        },
        method="POST",
    )
    payload = await asyncio.to_thread(_read_json_response, request, timeout)
    bearer = payload.get("gRPCToken")
    if not isinstance(bearer, str) or not bearer:
        raise MediaGenerationKitError.generation_failed(
            "cloud authentication response missing gRPCToken"
        )
    return bearer


def _fetch_payg_enabled(short_term_token: str, base_url: str, timeout: float) -> bool:
    cached = _cached_payg_enabled(short_term_token, base_url)
    if cached is not None:
        return cached

    request = urllib.request.Request(
        _join_url(base_url, "/billing/stripe/payg"),
        headers={
            "Authorization": short_term_token,
            "User-Agent": DEFAULT_HTTP_USER_AGENT,
        },
        method="GET",
    )
    try:
        payload = _read_json_response(request, timeout=timeout)
    except MediaGenerationKitError:
        _cache_payg_enabled(False, short_term_token, base_url)
        return False

    is_enabled = bool(payload.get("paygEnabled")) and bool(payload.get("paygEligible"))
    _cache_payg_enabled(is_enabled, short_term_token, base_url)
    return is_enabled


def _cached_payg_enabled(short_term_token: str, base_url: str) -> bool | None:
    key = (short_term_token, _normalize_base_url(base_url))
    now = time.monotonic()
    with _payg_cache_lock:
        cached = _payg_status_cache.get(key)
        if cached is None:
            return None
        enabled, cached_at = cached
        if now - cached_at > _PAYG_CACHE_TTL:
            _payg_status_cache.pop(key, None)
            return None
        return enabled


def _cache_payg_enabled(enabled: bool, short_term_token: str, base_url: str) -> None:
    key = (short_term_token, _normalize_base_url(base_url))
    with _payg_cache_lock:
        _payg_status_cache[key] = (enabled, time.monotonic())


def _read_json_response(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise MediaGenerationKitError.generation_failed(
            f"cloud request failed with status {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise MediaGenerationKitError.generation_failed(
            f"cloud request failed: {error.reason}"
        ) from error
    except Exception as error:
        raise MediaGenerationKitError.generation_failed(
            f"cloud request failed: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise MediaGenerationKitError.generation_failed("cloud response was not a JSON object")
    return payload


def _join_url(base_url: str, path: str) -> str:
    return f"{_normalize_base_url(base_url)}{path}"


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        {key: value for key, value in payload.items() if value is not None},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")
