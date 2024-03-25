load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "_CollectionsUtilities",
    srcs = glob([
        "Sources/_CollectionsUtilities/**/*.swift",
    ]),
    module_name = "_CollectionsUtilities",
)

swift_library(
    name = "BitCollections",
    srcs = glob([
        "Sources/BitCollections/**/*.swift",
    ]),
    module_name = "BitCollections",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "DequeModule",
    srcs = glob([
        "Sources/DequeModule/**/*.swift",
    ]),
    module_name = "DequeModule",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "HeapModule",
    srcs = glob([
        "Sources/HeapModule/**/*.swift",
    ]),
    module_name = "HeapModule",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "OrderedCollections",
    srcs = glob([
        "Sources/OrderedCollections/**/*.swift",
    ]),
    module_name = "OrderedCollections",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "PersistentCollections",
    srcs = glob([
        "Sources/PersistentCollections/**/*.swift",
    ]),
    module_name = "PersistentCollections",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "SortedCollections",
    srcs = glob([
        "Sources/SortedCollections/**/*.swift",
    ]),
    module_name = "SortedCollections",
    deps = [
        ":_CollectionsUtilities",
    ],
)

swift_library(
    name = "Collections",
    srcs = glob([
        "Sources/Collections/**/*.swift",
    ]),
    module_name = "Collections",
    visibility = ["//visibility:public"],
    deps = [
        ":BitCollections",
        ":DequeModule",
        ":HeapModule",
        ":OrderedCollections",
        ":PersistentCollections",
        ":SortedCollections",
    ],
)
