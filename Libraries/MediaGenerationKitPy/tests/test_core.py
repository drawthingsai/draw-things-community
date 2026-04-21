from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import unittest
from unittest import mock

import numpy as np

from media_generation_kit import (
    CloudComputeBackend,
    MediaGenerationEnvironment,
    MediaGenerationPipeline,
)
from media_generation_kit.catalog import recommended_configuration
from media_generation_kit.cloud import CloudAuthenticator, cloud_authenticate, resolve_api_key
from media_generation_kit.errors import MediaGenerationKitError
from media_generation_kit.flatbuffer import serialize_configuration
from media_generation_kit.image_codec import image_to_tensor_data
from media_generation_kit.proto import image_service_pb2 as pb2
from media_generation_kit.remote import RemoteExecutor
from media_generation_kit.tensor import (
    CCV_32F,
    FPZIP_IDENTIFIER,
    TENSOR_HEADER_SIZE,
    decode_tensor,
    encode_float_tensor_raw,
)


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p2ioAAAAASUVORK5CYII="
)


class MediaGenerationKitPyTests(unittest.TestCase):
    def test_environment_resolves_bundled_file(self) -> None:
        async def run():
            return await MediaGenerationEnvironment.default.resolve_model(
                "flux_2_klein_4b_f16.ckpt",
                offline=True,
            )

        resolved = asyncio.run(run())
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.file, "flux_2_klein_4b_f16.ckpt")

    def test_environment_resolveModel_alias(self) -> None:
        async def run():
            return await MediaGenerationEnvironment.default.resolveModel(
                "flux_2_klein_4b_f16.ckpt",
                offline=True,
            )

        resolved = asyncio.run(run())
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.file, "flux_2_klein_4b_f16.ckpt")

    def test_environment_resolves_display_name_without_variant_suffix(self) -> None:
        async def run():
            return await MediaGenerationEnvironment.default.resolve_model(
                "FLUX.2 [klein] 4B",
                offline=True,
            )

        resolved = asyncio.run(run())
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.file, "flux_2_klein_4b_f16.ckpt")

    def test_environment_resolves_hugging_face_reference(self) -> None:
        async def run():
            return await MediaGenerationEnvironment.default.resolve_model(
                "hf://black-forest-labs/FLUX.2-klein-4B",
                offline=True,
            )

        resolved = asyncio.run(run())
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.file, "flux_2_klein_4b_f16.ckpt")

    def test_recommended_configuration_uses_bundled_template(self) -> None:
        async def run():
            return await recommended_configuration("flux_2_klein_4b_f16.ckpt", offline=True)

        configuration = asyncio.run(run())
        self.assertEqual(configuration.model, "flux_2_klein_4b_f16.ckpt")
        self.assertEqual(configuration.width, 1024)
        self.assertEqual(configuration.height, 1024)
        self.assertEqual(configuration.steps, 4)

    def test_environment_downloadableModels_alias(self) -> None:
        async def run():
            return await MediaGenerationEnvironment.default.downloadableModels(offline=True)

        models = asyncio.run(run())
        self.assertTrue(any(model.file == "flux_2_klein_4b_f16.ckpt" for model in models))

    def test_flatbuffer_validation_rejects_invalid_dimensions(self) -> None:
        async def run():
            return await recommended_configuration("flux_2_klein_4b_f16.ckpt", offline=True)

        configuration = asyncio.run(run())
        configuration.width = 513
        with self.assertRaises(MediaGenerationKitError):
            serialize_configuration(configuration)

    def test_flatbuffer_serialization_returns_payload(self) -> None:
        async def run():
            return await recommended_configuration("flux_2_klein_4b_f16.ckpt", offline=True)

        payload = serialize_configuration(asyncio.run(run()))
        self.assertGreater(len(payload), 64)

    def test_tensor_raw_round_trip(self) -> None:
        payload = encode_float_tensor_raw(
            (1, 1, 2, 3),
            np.array([0, 0.5, 1, -1, -0.5, 0.25], dtype=np.float32),
        )
        tensor = decode_tensor(payload)
        self.assertEqual(tensor.shape, (1, 1, 2, 3))
        self.assertEqual(tensor.width, 2)
        self.assertEqual(tensor.height, 1)

    def test_tensor_fpzip_round_trip(self) -> None:
        import fpzip

        array = np.array([0, 0.5, 1, -1, -0.5, 0.25], dtype=np.float32).reshape((1, 1, 2, 3))
        payload = fpzip.compress(array, precision=0, order="C")

        header = bytearray(TENSOR_HEADER_SIZE)
        struct.pack_into("<I", header, 0, FPZIP_IDENTIFIER)
        struct.pack_into("<i", header, 4, 0x1)
        struct.pack_into("<i", header, 8, 0x02)
        struct.pack_into("<i", header, 12, CCV_32F)
        struct.pack_into("<i", header, 16, 0)
        for index, value in enumerate(array.shape):
            struct.pack_into("<i", header, 20 + index * 4, int(value))

        tensor = decode_tensor(bytes(header) + payload)
        self.assertEqual(tensor.shape, (1, 1, 2, 3))
        self.assertTrue(np.allclose(tensor.data.reshape(array.shape), array))

    def test_image_input_roles(self) -> None:
        image = MediaGenerationPipeline.data(ONE_PIXEL_PNG)
        self.assertEqual(image.role, "image")
        self.assertEqual(image.mask().role, "mask")
        self.assertEqual(image.moodboard().role, "moodboard")
        self.assertEqual(image.depth().role, "depth")

    def test_image_to_tensor_data(self) -> None:
        tensor_payload, auto_mask = image_to_tensor_data(ONE_PIXEL_PNG, 64, 64)
        tensor = decode_tensor(tensor_payload)
        self.assertEqual(tensor.shape, (1, 64, 64, 3))
        self.assertIsNone(auto_mask)

    def test_dynamic_proto_round_trip(self) -> None:
        request = pb2.EchoRequest(name="MediaGenerationKitPy", sharedSecret="secret")
        decoded = pb2.EchoRequest.FromString(request.SerializeToString())
        self.assertEqual(decoded.name, "MediaGenerationKitPy")
        self.assertEqual(decoded.sharedSecret, "secret")

    def test_dynamic_proto_signpost_round_trip(self) -> None:
        signpost = pb2.ImageGenerationSignpostProto()
        signpost.sampling.step = 2
        response = pb2.ImageGenerationResponse(currentSignpost=signpost, chunkState=pb2.LAST_CHUNK)
        decoded = pb2.ImageGenerationResponse.FromString(response.SerializeToString())
        self.assertTrue(decoded.HasField("currentSignpost"))
        self.assertEqual(decoded.currentSignpost.WhichOneof("signpost"), "sampling")
        self.assertEqual(decoded.currentSignpost.sampling.step, 2)

    def test_pipeline_async_context_closes_executor(self) -> None:
        class Executor:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        async def run() -> Executor:
            configuration = await recommended_configuration(
                "flux_2_klein_4b_f16.ckpt",
                offline=True,
            )
            executor = Executor()
            async with MediaGenerationPipeline(
                backend=object(),
                configuration=configuration,
                model=configuration.model,
                remote_executor=executor,
            ):
                self.assertFalse(executor.closed)
            return executor

        self.assertTrue(asyncio.run(run()).closed)

    def test_resolve_api_key_uses_environment(self) -> None:
        with mock.patch.dict(os.environ, {"DRAWTHINGS_API_KEY": "dk_env"}, clear=True):
            self.assertEqual(resolve_api_key(None), "dk_env")

    def test_cloud_authenticator_short_term_token_caches_response(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                del exc_type, exc, tb
                return False

        def fake_urlopen(request, timeout=0):
            del timeout
            calls.append(
                {
                    "url": request.full_url,
                    "body": json.loads(request.data.decode("utf-8")),
                    "headers": {key.lower(): value for key, value in request.header_items()},
                }
            )
            return FakeResponse({"shortTermToken": "st_123", "expiresIn": 3600})

        async def run() -> tuple[str, str]:
            authenticator = CloudAuthenticator("dk_test")
            first = await authenticator.short_term_token()
            second = await authenticator.short_term_token()
            return first, second

        with mock.patch("media_generation_kit.cloud.urllib.request.urlopen", side_effect=fake_urlopen):
            first, second = asyncio.run(run())

        self.assertEqual(first, "st_123")
        self.assertEqual(second, "st_123")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], "https://api.drawthings.ai/sdk/token")
        self.assertEqual(calls[0]["body"]["apiKey"], "dk_test")
        self.assertEqual(calls[0]["body"]["appCheckType"], "none")
        self.assertEqual(calls[0]["headers"]["user-agent"], "MediaGenerationKitPy/0.1")

    def test_cloud_authenticate_returns_bearer_token(self) -> None:
        requests: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                del exc_type, exc, tb
                return False

        def fake_urlopen(request, timeout=0):
            del timeout
            headers = {key.lower(): value for key, value in request.header_items()}
            body = json.loads(request.data.decode("utf-8")) if request.data else None
            requests.append({"url": request.full_url, "headers": headers, "body": body})
            if request.full_url.endswith("/billing/stripe/payg"):
                return FakeResponse({"paygEnabled": True, "paygEligible": True})
            if request.full_url.endswith("/authenticate"):
                return FakeResponse({"gRPCToken": "grpc-token"})
            raise AssertionError(f"unexpected URL {request.full_url}")

        async def run() -> str:
            return await cloud_authenticate(
                short_term_token="short-term-token",
                encoded_blob="ZW5jb2RlZA==",
                from_bridge=True,
                estimated_compute_units=12.0,
            )

        with mock.patch("media_generation_kit.cloud.urllib.request.urlopen", side_effect=fake_urlopen):
            token = asyncio.run(run())

        self.assertEqual(token, "grpc-token")
        self.assertEqual(requests[0]["url"], "https://api.drawthings.ai/billing/stripe/payg")
        self.assertEqual(requests[1]["url"], "https://api.drawthings.ai/authenticate")
        self.assertEqual(requests[1]["headers"]["authorization"], "short-term-token")
        self.assertEqual(requests[1]["body"]["blob"], "ZW5jb2RlZA==")
        self.assertEqual(requests[1]["body"]["consumableType"], "payg")
        self.assertEqual(requests[1]["body"]["amount"], 12.0)

    def test_pipeline_accepts_cloud_compute_backend(self) -> None:
        class Executor:
            async def prepare_for_generation(self) -> None:
                return None

            async def generate(self, prompt, negative_prompt, configuration, inputs, state_handler):
                del prompt, negative_prompt, configuration, inputs, state_handler
                return []

            async def aclose(self) -> None:
                return None

        async def run():
            with mock.patch(
                "media_generation_kit.pipeline.RemoteExecutor.create",
                new=mock.AsyncMock(return_value=Executor()),
            ):
                return await MediaGenerationPipeline.from_pretrained(
                    "flux_2_klein_4b_f16.ckpt",
                    backend=CloudComputeBackend(api_key="dk_test"),
                )

        pipeline = asyncio.run(run())
        self.assertEqual(pipeline.model, "flux_2_klein_4b_f16.ckpt")

    def test_request_includes_model_override_metadata(self) -> None:
        configuration = asyncio.run(
            recommended_configuration("flux_2_klein_4b_f16.ckpt", offline=True)
        )

        class Stub:
            pass

        executor = RemoteExecutor(
            CloudComputeBackend(api_key="dk_test"),
            channel=object(),
            stub=Stub(),
            cloud_authenticator=None,
        )

        async def run():
            prepared = await executor._build_request(  # noqa: SLF001
                "a cat in studio lighting",
                "",
                configuration,
                [],
                None,
            )
            return json.loads(prepared.request.override.models.decode("utf-8"))

        models = asyncio.run(run())
        self.assertTrue(models)
        self.assertEqual(models[0]["file"], "flux_2_klein_4b_f16.ckpt")
        self.assertEqual(models[0]["modifier"], "kontext")

    def test_remote_executor_cloud_uses_bearer_metadata(self) -> None:
        class FakeCall:
            def __init__(self, responses):
                self._responses = iter(responses)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._responses)
                except StopIteration as error:
                    raise StopAsyncIteration from error

            def cancel(self) -> None:
                return None

        class FakeStub:
            def __init__(self, response):
                self.response = response
                self.metadata = None

            def GenerateImage(self, request, metadata=None):
                del request
                self.metadata = metadata
                return FakeCall([self.response])

        class FakeAuthenticator:
            def __init__(self) -> None:
                self.base_url = "https://api.drawthings.ai"
                self._remote_models = {"flux_2_klein_4b_f16.ckpt"}

            async def short_term_token(self):
                return "short-term-token"

            def remote_models(self) -> set[str]:
                return set(self._remote_models)

            def update_remote_models_from_handshake(self, models) -> None:
                self._remote_models = set(models)

        configuration = asyncio.run(
            recommended_configuration("flux_2_klein_4b_f16.ckpt", offline=True)
        )
        payload = encode_float_tensor_raw((1, 1, 1, 3), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        response = pb2.ImageGenerationResponse(generatedImages=[payload], chunkState=pb2.LAST_CHUNK)
        stub = FakeStub(response)
        executor = RemoteExecutor(
            CloudComputeBackend(api_key="dk_test"),
            channel=object(),
            stub=stub,
            cloud_authenticator=FakeAuthenticator(),
        )

        async def run() -> None:
            with mock.patch(
                "media_generation_kit.remote.cloud_authenticate",
                new=mock.AsyncMock(return_value="grpc-token"),
            ):
                results = await executor.generate("a cube", "", configuration, [])
                self.assertEqual(len(results), 1)

        asyncio.run(run())
        self.assertEqual(stub.metadata, (("authorization", "bearer grpc-token"),))


if __name__ == "__main__":
    unittest.main()
