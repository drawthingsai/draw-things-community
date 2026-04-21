from __future__ import annotations

from . import image_service_pb2 as pb2


class ImageGenerationServiceStub:
    def __init__(self, channel):
        try:
            import grpc
        except ModuleNotFoundError as error:
            raise RuntimeError("grpcio is required for remote generation") from error

        self.Echo = channel.unary_unary(
            "/ImageGenerationService/Echo",
            request_serializer=pb2.EchoRequest.SerializeToString,
            response_deserializer=pb2.EchoReply.FromString,
        )
        self.GenerateImage = channel.unary_stream(
            "/ImageGenerationService/GenerateImage",
            request_serializer=pb2.ImageGenerationRequest.SerializeToString,
            response_deserializer=pb2.ImageGenerationResponse.FromString,
        )
        del grpc
