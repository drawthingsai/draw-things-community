# PDF Markdown extraction

Native core from [firecrawl/pdf-inspector](https://github.com/firecrawl/pdf-inspector),
pinned to `2dbd16b3ea4ea943af2802bd72e45dd264eacb9f` (1.25.0, MIT).
Bazel builds Rust into a static library; `PDFTextExtractor` copies its synchronous
per-page callbacks into Swift values. No Python, PDFium, OCR models, or remote
service is used.

`password.patch` exposes the existing password-aware per-page implementation.
The normal whole-document Markdown API's page markers do not reliably cover
table-only pages, so Read adds its own `<!-- Page N -->` markers. Pages flagged
as needing OCR get an explicit placeholder and can be inspected with View.

Read applies the existing line pagination and `LINE:TAG>` formatting to the
virtual Markdown. Its continuation retains the Markdown, resolved URL, and source
SHA-256, never the password. More checks live permissions and source bytes before
reusing it. Completing a read clears the continuation. Edit rejects PDFs.
PDFKit validates passwords through the existing interactive password hook before
native extraction, so unsupported extractor encryption does not cause retry loops.

To update dependencies, change `Cargo.toml`, regenerate `Cargo.lock`, and run
`CARGO_BAZEL_REPIN=1 bazel sync --only=pdf_inspector_crates`. Keep the revision and
archive checksum in `repositories.bzl` and `build-swiftpm.sh` in sync.

For the diagnostic SwiftPM Catalyst app, `build-swiftpm.sh` generates the native
archive separately; see `Apps/LocalCodeCatalyst/README.md`. The script uses the
same dependency lockfile and native-only feature set as Bazel.
