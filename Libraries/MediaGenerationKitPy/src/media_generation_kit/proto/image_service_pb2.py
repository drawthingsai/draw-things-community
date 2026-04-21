from __future__ import annotations

from google.protobuf import descriptor_pb2, message_factory


def _field(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    label: int = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    type_name: str | None = None,
    oneof_index: int | None = None,
) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name is not None:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _message(fd: descriptor_pb2.FileDescriptorProto, name: str) -> descriptor_pb2.DescriptorProto:
    message = fd.message_type.add()
    message.name = name
    return message


def _enum(
    fd: descriptor_pb2.FileDescriptorProto,
    name: str,
    values: list[tuple[str, int]],
) -> None:
    enum = fd.enum_type.add()
    enum.name = name
    for value_name, number in values:
        value = enum.value.add()
        value.name = value_name
        value.number = number


def _build_file_descriptor() -> descriptor_pb2.FileDescriptorProto:
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = "imageService.proto"
    fd.syntax = "proto3"

    _enum(fd, "DeviceType", [("PHONE", 0), ("TABLET", 1), ("LAPTOP", 2)])
    _enum(fd, "ChunkState", [("LAST_CHUNK", 0), ("MORE_CHUNKS", 1)])

    message = _message(fd, "EchoRequest")
    _field(message, "name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(message, "sharedSecret", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    message = _message(fd, "ComputeUnitThreshold")
    _field(message, "community", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _field(message, "plus", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _field(message, "expireAt", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)

    message = _message(fd, "MetadataOverride")
    for index, name in enumerate(["models", "loras", "controlNets", "textualInversions", "upscalers"], 1):
        _field(message, name, index, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)

    message = _message(fd, "EchoReply")
    _field(message, "message", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(
        message,
        "files",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(message, "override", 3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=".MetadataOverride")
    _field(message, "sharedSecretMissing", 4, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    _field(
        message,
        "thresholds",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ComputeUnitThreshold",
    )
    _field(message, "serverIdentifier", 6, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)

    message = _message(fd, "FileListRequest")
    _field(
        message,
        "files",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(
        message,
        "filesWithHash",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(message, "sharedSecret", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    message = _message(fd, "FileExistenceResponse")
    _field(
        message,
        "files",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(
        message,
        "existences",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(
        message,
        "hashes",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )

    message = _message(fd, "TensorAndWeight")
    _field(message, "tensor", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    _field(message, "weight", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)

    message = _message(fd, "HintProto")
    _field(message, "hintType", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(
        message,
        "tensors",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=".TensorAndWeight",
    )

    message = _message(fd, "ImageGenerationRequest")
    _field(message, "image", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    _field(message, "scaleFactor", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(message, "mask", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    _field(
        message,
        "hints",
        4,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=".HintProto",
    )
    _field(message, "prompt", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(message, "negativePrompt", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(message, "configuration", 7, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    _field(message, "override", 8, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=".MetadataOverride")
    _field(
        message,
        "keywords",
        9,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(message, "user", 10, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(message, "device", 11, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, type_name=".DeviceType")
    _field(
        message,
        "contents",
        12,
        descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(message, "sharedSecret", 13, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(message, "chunked", 14, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)

    message = _message(fd, "ImageGenerationSignpostProto")
    message.oneof_decl.add().name = "signpost"
    for nested_name in [
        "TextEncoded",
        "ImageEncoded",
        "ImageDecoded",
        "SecondPassImageEncoded",
        "SecondPassImageDecoded",
        "FaceRestored",
        "ImageUpscaled",
    ]:
        message.nested_type.add().name = nested_name
    nested = message.nested_type.add()
    nested.name = "Sampling"
    _field(nested, "step", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    nested = message.nested_type.add()
    nested.name = "SecondPassSampling"
    _field(nested, "step", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(
        message,
        "textEncoded",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.TextEncoded",
        oneof_index=0,
    )
    _field(
        message,
        "imageEncoded",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.ImageEncoded",
        oneof_index=0,
    )
    _field(
        message,
        "sampling",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.Sampling",
        oneof_index=0,
    )
    _field(
        message,
        "imageDecoded",
        4,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.ImageDecoded",
        oneof_index=0,
    )
    _field(
        message,
        "secondPassImageEncoded",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.SecondPassImageEncoded",
        oneof_index=0,
    )
    _field(
        message,
        "secondPassSampling",
        6,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.SecondPassSampling",
        oneof_index=0,
    )
    _field(
        message,
        "secondPassImageDecoded",
        7,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.SecondPassImageDecoded",
        oneof_index=0,
    )
    _field(
        message,
        "faceRestored",
        8,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.FaceRestored",
        oneof_index=0,
    )
    _field(
        message,
        "imageUpscaled",
        9,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto.ImageUpscaled",
        oneof_index=0,
    )

    message = _message(fd, "RemoteDownloadResponse")
    _field(message, "bytesReceived", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _field(message, "bytesExpected", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _field(message, "item", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(message, "itemsExpected", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(message, "tag", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    message = _message(fd, "ImageGenerationResponse")
    _field(
        message,
        "generatedImages",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(
        message,
        "currentSignpost",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".ImageGenerationSignpostProto",
    )
    _field(
        message,
        "signposts",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=".ImageGenerationSignpostProto",
    )
    _field(message, "previewImage", 4, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    _field(message, "scaleFactor", 5, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
    _field(
        message,
        "tags",
        6,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _field(message, "downloadSize", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _field(message, "chunkState", 8, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, type_name=".ChunkState")
    _field(
        message,
        "remoteDownload",
        9,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".RemoteDownloadResponse",
    )
    _field(
        message,
        "generatedAudio",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )

    message = _message(fd, "HoursRequest")
    message = _message(fd, "HoursResponse")
    _field(message, "thresholds", 1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=".ComputeUnitThreshold")

    return fd


_MESSAGES = message_factory.GetMessages([_build_file_descriptor()])

EchoRequest = _MESSAGES["EchoRequest"]
EchoReply = _MESSAGES["EchoReply"]
MetadataOverride = _MESSAGES["MetadataOverride"]
ImageGenerationRequest = _MESSAGES["ImageGenerationRequest"]
ImageGenerationResponse = _MESSAGES["ImageGenerationResponse"]
HintProto = _MESSAGES["HintProto"]
TensorAndWeight = _MESSAGES["TensorAndWeight"]
ImageGenerationSignpostProto = _MESSAGES["ImageGenerationSignpostProto"]
RemoteDownloadResponse = _MESSAGES["RemoteDownloadResponse"]

PHONE = 0
TABLET = 1
LAPTOP = 2
LAST_CHUNK = 0
MORE_CHUNKS = 1
