# OSH for Local Code

Local Code embeds the native C++ OSH runtime from Oils 0.37.0. The upstream
release is pinned by SHA-256 in the workspace files. `osh-ios.patch` adds a
small no-process `osh_ios` shim and C callback API; OSH does not depend on
`ios_system` directly.

The embedded mode supports normal OSH syntax, functions, indexed and
associative arrays, arithmetic, `[[ ... ]]`, command substitution, subshells,
finite pipelines, here-documents, redirects on descriptors 0/1/2, and exact
argument-vector external commands. Subshells, command substitutions, and each
pipeline stage clone mutable shell state, options, functions, and traps, then
restore their working directory. A virtual child runs its own `EXIT` trap and
cannot replace the parent's trap table.

External commands write directly to file-backed stdout/stderr targets, so a
long-running command exposes readiness and progress before it exits. Command
substitution still captures its output.

The streaming prototype runs pipelines of literal external commands concurrently
using invocation-owned pipes and separate native command sessions. For example,
`hyperframes preview --foreground --no-open 2>&1 | tee preview.log` exposes the
URL in both Bash output and the log while preview runs. It preserves ordered
redirects on descriptors 0/1/2, stage exit statuses, and `pipefail`; Bash
cancellation reaches every stage. Consumers must be registered thread-safe
commands (`tee` now uses invocation-local state and unbuffered output files).
The existing Node queue classification is unchanged.

Pipelines with expansions, shell builtins/functions, compound stages, active
traps, or consumers requiring exclusive execution keep the finite buffered
path. Those buffered pipelines have a 256 MiB per-stage limit checked after
each intermediate stage completes. Exceeding the limit emits a diagnostic,
skips the remaining stages, and fails the pipeline
with status 125 without throwing; scripts can handle the failure with `||` or
`$?`. A background job requested with `&` emits a warning and runs
synchronously in isolated shell state inside the current Local Code Bash job.
It does not create a shell job, populate `$!`, or outlive that Bash job.
Process substitution, named descriptors, and other descriptor manipulation are
reported as unsupported and return status 125. `exec` invokes the exact-argument
external callback and exits only the current virtual shell context. Local Code
registers the OSH invocation and its external commands with ios_system virtual
PIDs. The current PID is thread-local and explicitly inherited by each native
command thread; the shared PID-slot registry is atomic. Foreground commands join
through cleanup before releasing their streams, including when the host normally
launches commands asynchronously. `$$`, signal registration, and signals sent to
the virtual shell stay in its invocation context; signals for external commands
route through `ios_killpid`. OSH never replaces a process-wide signal disposition. Other
native builtins remain linked to ios_system's libc replacement layer.
Unsupported process syntax must never fall through to real `fork`, `execve`,
or process-global `dup2`. Streaming uses only explicitly owned pipe descriptors.

Each OSH worker owns a thread-local mycpp heap. Heap-owned runtime globals,
including the grammar cache, standard-stream wrappers, readline state, and
signal state, are thread-local as well. Per-invocation callback, virtual-input,
redirect, and cancellation state remains thread-local. This lets an evaluator
stay suspended in a long-running external callback while another evaluator
runs on a different worker without sharing a collector or root stack. The
thread-local heap releases its objects when that worker exits.
