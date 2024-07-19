load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@dflat//:dflat.bzl", "dflatc")

dflatc(
    name = "text_history_schema",
    srcs = [
        "Sources/text_history.fbs",
        "Sources/text_lineage.fbs",
    ],
)

dflatc(
    name = "tensor_history_schema",
    srcs = [
        "Sources/script_session.fbs",
        "Sources/tensor_data.fbs",
        "Sources/tensor_history.fbs",
        "Sources/tensor_moodboard_data.fbs",
    ],
)

dflatc(
    name = "thumbnail_history_schema",
    srcs = [
        "Sources/thumbnail_history.fbs",
        "Sources/thumbnail_history_half.fbs",
    ],
)

swift_library(
    name = "History",
    srcs = glob(["Sources/**/*.swift"]) + [
        ":text_history_schema",
        ":tensor_history_schema",
        ":thumbnail_history_schema",
    ],
    module_name = "History",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/SwiftDiffusion",
        "@dflat//:SQLiteDflat",
        "@s4nnc//nnc",
    ],
)
