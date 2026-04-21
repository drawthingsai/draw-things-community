from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .backend import CloudComputeBackend, RemoteBackend
from .catalog import metadata_override_models
from .cloud import (
    CloudAuthenticator,
    CloudAuthenticatorRegistry,
    DEFAULT_CLOUD_COMPUTE_HOST,
    DEFAULT_CLOUD_COMPUTE_PORT,
    cloud_authenticate,
    prefetch_payg_enabled,
    resolve_api_key,
)
from .configuration import Configuration
from .errors import MediaGenerationKitError
from .flatbuffer import serialize_configuration
from .image_codec import hint_to_tensor_data, image_to_tensor_data, mask_to_tensor_data
from .input import MediaGenerationInput
from .proto import image_service_pb2 as pb2
from .proto.image_service_pb2_grpc import ImageGenerationServiceStub
from .result import Preview, Result
from .states import (
    Decoding,
    Downloading,
    EncodingInputs,
    EncodingText,
    Generating,
    PipelineState,
    Postprocessing,
    Uploading,
)
from .tensor import decode_tensor

StateHandler = Callable[[PipelineState, Preview | None], None]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    request: Any
    encoded_without_contents: bytes
    has_image: bool
    shuffle_count: int


class RemoteExecutor:
    def __init__(
        self,
        backend: RemoteBackend | CloudComputeBackend,
        channel: Any,
        stub: ImageGenerationServiceStub,
        *,
        cloud_authenticator: CloudAuthenticator | None = None,
    ):
        self.backend = backend
        self._channel = channel
        self._stub = stub
        self._cloud_authenticator = cloud_authenticator
        self.server_identifier: int = 0
        self.discovered_remote_models: set[str] = set()

    @classmethod
    async def create(cls, backend: RemoteBackend | CloudComputeBackend) -> "RemoteExecutor":
        grpc = _import_grpc()
        options = [
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ]
        cloud_authenticator = None
        if isinstance(backend, CloudComputeBackend):
            target = f"{DEFAULT_CLOUD_COMPUTE_HOST}:{DEFAULT_CLOUD_COMPUTE_PORT}"
            channel = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials(), options=options)
            cloud_authenticator = CloudAuthenticatorRegistry.shared.authenticator(  # type: ignore[attr-defined]
                resolve_api_key(backend.api_key),
                base_url=backend.base_url,
            )
        else:
            target = f"{backend.host}:{backend.port}"
            if backend.use_tls:
                channel = grpc.aio.secure_channel(
                    target, grpc.ssl_channel_credentials(), options=options
                )
            else:
                channel = grpc.aio.insecure_channel(target, options=options)
        executor = cls(
            backend,
            channel,
            ImageGenerationServiceStub(channel),
            cloud_authenticator=cloud_authenticator,
        )
        await executor.handshake()
        return executor

    async def aclose(self) -> None:
        close = getattr(self._channel, "close", None)
        if close is not None:
            await close()

    async def prepare_for_generation(self) -> None:
        if self._cloud_authenticator is None:
            return
        short_term_token = await self._cloud_authenticator.short_term_token()
        await prefetch_payg_enabled(
            short_term_token,
            base_url=self._cloud_authenticator.base_url,
        )

    async def handshake(self) -> set[str]:
        request = pb2.EchoRequest(name=self.backend.device_name)
        if isinstance(self.backend, RemoteBackend) and self.backend.shared_secret:
            request.sharedSecret = self.backend.shared_secret
        response = await self._stub.Echo(request, timeout=5)
        self.server_identifier = int(getattr(response, "serverIdentifier", 0) or 0)
        models = set(getattr(response, "files", []) or [])
        override = getattr(response, "override", None)
        if override is not None and getattr(override, "models", b""):
            models.update(_decode_override_model_files(override.models))
        self.discovered_remote_models = models
        if self._cloud_authenticator is not None:
            self._cloud_authenticator.update_remote_models_from_handshake(models)
        return models

    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        configuration: Configuration,
        inputs: list[MediaGenerationInput],
        state_handler: StateHandler | None = None,
    ) -> list[Result]:
        if self._cloud_authenticator is not None:
            remote_models = self._cloud_authenticator.remote_models()
            if configuration.model and configuration.model not in remote_models:
                raise MediaGenerationKitError.model_not_found_on_remote(configuration.model)

        prepared_request = await self._build_request(
            prompt,
            negative_prompt,
            configuration,
            inputs,
            state_handler,
        )
        metadata = None
        if self._cloud_authenticator is not None:
            short_term_token = await self._cloud_authenticator.short_term_token()
            bearer = await cloud_authenticate(
                short_term_token=short_term_token,
                encoded_blob=base64.b64encode(prepared_request.encoded_without_contents).decode("ascii"),
                from_bridge=True,
                estimated_compute_units=_estimated_compute_units(
                    configuration,
                    has_image=prepared_request.has_image,
                    shuffle_count=prepared_request.shuffle_count,
                ),
                base_url=self._cloud_authenticator.base_url,
                timeout=30.0,
            )
            metadata = (("authorization", f"bearer {bearer}"),)
        tensors = []
        image_chunks: list[bytes] = []
        expected_download = 0
        received_download = 0

        if metadata is None:
            call = self._stub.GenerateImage(prepared_request.request)
        else:
            call = self._stub.GenerateImage(prepared_request.request, metadata=metadata)
        try:
            async for response in call:
                remote_download = getattr(response, "remoteDownload", None)
                if remote_download is not None and getattr(remote_download, "bytesExpected", 0):
                    state_handler and state_handler(
                        Downloading(
                            bytes_received=int(remote_download.bytesReceived),
                            total_bytes=int(remote_download.bytesExpected),
                        ),
                        None,
                    )

                if getattr(response, "downloadSize", 0):
                    expected_download = int(response.downloadSize)
                    state_handler and state_handler(Downloading(0, expected_download), None)

                if _has_field(response, "currentSignpost"):
                    preview = None
                    if getattr(response, "previewImage", b""):
                        try:
                            preview = Preview([decode_tensor(response.previewImage)])
                        except MediaGenerationKitError:
                            preview = None
                    state_handler and state_handler(
                        _state_from_signpost(response.currentSignpost, configuration.steps),
                        preview,
                    )

                for image in getattr(response, "generatedImages", []) or []:
                    image_bytes = bytes(image)
                    received_download += len(image_bytes)
                    if expected_download > 0:
                        state_handler and state_handler(
                            Downloading(received_download, expected_download),
                            None,
                        )
                    if int(getattr(response, "chunkState", pb2.LAST_CHUNK)) == pb2.MORE_CHUNKS:
                        image_chunks.append(image_bytes)
                    else:
                        payload = b"".join([*image_chunks, image_bytes]) if image_chunks else image_bytes
                        image_chunks.clear()
                        tensors.append(decode_tensor(payload))
        except asyncio.CancelledError:
            cancel = getattr(call, "cancel", None)
            if cancel is not None:
                cancel()
            raise

        if not tensors:
            raise MediaGenerationKitError.generation_failed()
        return [Result(tensor) for tensor in tensors]

    async def _build_request(
        self,
        prompt: str,
        negative_prompt: str,
        configuration: Configuration,
        inputs: list[MediaGenerationInput],
        state_handler: StateHandler | None,
    ) -> PreparedRequest:
        state_handler and state_handler(EncodingInputs(), None)
        contents: OrderedDict[bytes, bytes] = OrderedDict()
        image_hash: bytes | None = None
        mask_hash: bytes | None = None
        hints = []
        target_width = configuration.width
        target_height = configuration.height
        has_image = False
        shuffle_count = 0

        for input_value in inputs:
            data = await input_value.encoded_data()
            if input_value.role == "image":
                has_image = True
                if image_hash is not None:
                    raise MediaGenerationKitError.generation_failed(
                        "only one primary image input is supported"
                    )
                tensor_data, auto_mask = image_to_tensor_data(data, target_width, target_height)
                image_hash = _insert_content(contents, tensor_data)
                if auto_mask is not None and mask_hash is None:
                    mask_hash = _insert_content(contents, auto_mask)
            elif input_value.role == "mask":
                if mask_hash is not None:
                    raise MediaGenerationKitError.generation_failed("only one mask input is supported")
                mask_hash = _insert_content(
                    contents,
                    mask_to_tensor_data(data, target_width, target_height),
                )
            else:
                hint_type = "depth" if input_value.role == "depth" else "shuffle"
                if hint_type == "shuffle":
                    shuffle_count += 1
                tensor_hash = _insert_content(contents, hint_to_tensor_data(data))
                hints.append(
                    pb2.HintProto(
                        hintType=hint_type,
                        tensors=[pb2.TensorAndWeight(tensor=tensor_hash, weight=1.0)],
                    )
                )

        if (
            configuration.strength == 1
            and not configuration.controls
            and mask_hash is None
            and not hints
        ):
            image_hash = None
            contents.clear()

        serialized_configuration = serialize_configuration(configuration)
        model_overrides = await metadata_override_models(
            configuration.model,
            refiner_model=configuration.refiner_model,
            offline=False,
        )
        request = pb2.ImageGenerationRequest(
            image=image_hash or b"",
            scaleFactor=1,
            mask=mask_hash or b"",
            hints=hints,
            prompt=prompt,
            negativePrompt=negative_prompt,
            configuration=serialized_configuration,
            override=pb2.MetadataOverride(
                models=json.dumps(model_overrides, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ),
            keywords=[],
            user=self.backend.device_name,
            device=pb2.LAPTOP,
            contents=[],
            chunked=True,
        )
        if isinstance(self.backend, RemoteBackend) and self.backend.shared_secret:
            request.sharedSecret = self.backend.shared_secret

        encoded_without_contents = request.SerializeToString()
        request.contents.extend(contents.values())
        total_upload = len(encoded_without_contents) + sum(len(content) for content in contents.values())
        if total_upload > 0:
            state_handler and state_handler(Uploading(0, total_upload), None)
        return PreparedRequest(
            request=request,
            encoded_without_contents=encoded_without_contents,
            has_image=has_image,
            shuffle_count=shuffle_count,
        )


def _import_grpc():
    try:
        import grpc
    except ModuleNotFoundError as error:
        raise MediaGenerationKitError.generation_failed(
            "remote generation requires grpcio; install media-generation-kit with its dependencies"
        ) from error
    return grpc


def _insert_content(contents: OrderedDict[bytes, bytes], data: bytes) -> bytes:
    digest = hashlib.sha256(data).digest()
    contents[digest] = data
    return digest


def _state_from_signpost(signpost, total_steps: int) -> PipelineState:
    active = signpost.WhichOneof("signpost")
    if active == "textEncoded":
        return EncodingText()
    if active in {"imageEncoded", "secondPassImageEncoded"}:
        return EncodingInputs()
    if active == "sampling":
        step = int(signpost.sampling.step)
        return Generating(
            step=min(max(step + 1, 1), max(total_steps, 1)),
            total_steps=total_steps,
        )
    if active == "imageDecoded":
        return Decoding()
    return Postprocessing()


def _has_field(message, field: str) -> bool:
    try:
        return message.HasField(field)
    except ValueError:
        return bool(getattr(message, field, None))


def _decode_override_model_files(data: bytes) -> list[str]:
    import json

    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item["file"]) for item in payload if isinstance(item, dict) and item.get("file")]


def _estimated_compute_units(
    configuration: Configuration,
    *,
    has_image: bool,
    shuffle_count: int,
) -> float | None:
    megapixels = (configuration.width * configuration.height) / (1024 * 1024)
    reference_factor = 1 + max(shuffle_count, 0) * 0.15
    image_factor = max(configuration.strength, 0.05) if has_image else 1.0
    return float(
        max(
            1,
            math.ceil(
                megapixels
                * configuration.steps
                * configuration.batch_size
                * configuration.batch_count
                * image_factor
                * reference_factor
                * 12
            ),
        )
    )
