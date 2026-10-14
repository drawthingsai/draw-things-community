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

`osh_ios_run` takes a complete shell invocation. The `sh` and `bash` commands
(also `/bin/sh` and `/bin/bash`, and the matching shebangs) pass their argv to
OSH unchanged, so OSH parses options such as `-e`, `-u`, `-o pipefail`, and
`-l`, as well as `-c` with its `$0`, a script operand, and positional
parameters. With neither `-c` nor a script, the script is read from the
command's input one line at a time, so `read` and external commands in it
consume the input that follows, as with a native shell reading from a pipe.
Input is treated like a seekable file. An external command consumes only what
it reads, and the rest goes back to the shell. That applies whether the
command reads through its stdin stream (`ftell`) or its descriptor. A streaming
pipeline's first stage consumes what it reads from the descriptor, including
stdio read-ahead. OSH still reads a shell's whole input before it starts. When
that input is seekable, which it is for nested shells, it is moved back to just
past what the shell used when it finishes. So `sh`, `bash`, and scripts inside
a piped script consume only what they read.
`bash` selects OSH's bash personality, which sets
`BASH_VERSION`; it is OSH, not GNU Bash. Interactive shells (`-i`) return
status 125, and OSH 0.37 rejects `-s`.

Shells nest. `sh script`, `$(bash -c ...)`, and Python's `os.system` or
`subprocess.run(["sh", ...])` start another evaluator on their own command
thread while the enclosing one waits. They share the session's shell state,
which is saved and restored like a stack. Stopping a shell job sets a
session-wide cancellation flag before the foreground command is interrupted.
Every evaluator checks that flag before its next command, so the whole job
stops, nested shells included. A stop that arrives after an sh/bash command
starts but before its shell is running is picked up when that shell starts.
sh/bash command threads are never pthread-cancelled. Native callers such as
awk's `system()` run on pthread-cancellable threads. `ios_system_osh` defers
their cancellation until the shell finishes, then honours it.

Shell recursion runs on the C stack of the host process, roughly 5 KiB per
shell function call. Evaluators run on 16 MiB stacks: sh/bash command threads
get one from the command registry, and `ios_system_osh` moves to a helper thread
when its caller's stack is smaller, as with the Bash tool's queue threads and
Python. Runaway recursion stops with `osh: maximum recursion depth exceeded`
and status 2 while 256 KiB of stack remain, instead of overflowing the stack
and crashing the host.

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
