# Phase 0 security boundaries

## Assumptions

The application chooses a trusted workspace root and write capability. The root
is canonicalized once and pinned by an open directory descriptor. Relative paths
and file content may be untrusted. Calls on one instance are synchronous; callers
must not close it concurrently with I/O. The workspace is not an execution sandbox.

## Enforced behavior

- Absolute paths, `..` components, empty/noncanonical paths, NUL bytes, and
  backslash separators are rejected; no home expansion or URL interpretation.
- `.git`, `.venv`, `.env`, and `.env.*` components are protected, case-insensitively.
  This is a small built-in policy, not a complete secret-file discovery system.
- Every path component below the root is opened relative to a directory descriptor
  with `O_NOFOLLOW`. Leaf descriptors are checked again after opening. Symlinks
  are rejected even if they point inside the workspace; dangling links also fail.
- Only regular files with one hard link are accepted. Directories, FIFOs, and
  multiply linked files are refused. Reads use `O_NONBLOCK` before checking the
  opened file type so a substituted FIFO cannot block the reader.
- Reads are capped at `max_bytes + 1`, including when files grow after `stat`.
  UTF-8 bytes, not character counts, determine read/write size limits.
- Writes require `allow_write=True`. Existing destinations require
  `overwrite=True`. Parent directories are never implicitly created.
- Writes use an exclusive temporary file in the pinned parent directory, mode
  `0600`, followed by atomic publication. Creating uses `link` to refuse a competing
  destination; overwriting uses `replace` so it does not follow a substituted leaf
  symlink or modify the content of a destination hard link.
- Temporary files are cleaned up on ordinary success and failure. Invalid input
  is rejected before creating files. Errors expose stable codes and generic
  messages rather than absolute host paths.
- Provider and tool modules have no network clients, shell tools, or API adapters.
  Unit tests deny socket connections and use temporary workspaces.

## Limits

This protects access through these APIs. It cannot confine arbitrary Python,
shell commands, plugins, or other processes with the same OS permissions. An
adversarial process with access to the workspace can rename pinned directories
outside it or modify files concurrently. Descriptor traversal prevents symlink
redirection, but does not solve arbitrary external filesystem mutation, mounts,
or all platform-specific alias mechanisms. The trusted root must not contain
attacker-controlled mounts. Strong process isolation belongs to a later design.

Atomic replacement creates a new `0600` file and does not preserve an existing
file's mode, owner, ACLs, or extended attributes. Parent-directory write permission
controls replacement on POSIX; a read-only target can still be atomically replaced
when its parent is writable. A crash may leave a temporary file; directory fsync
and crash/power-loss durability are not provided. These tools target bounded text
files, not databases or general filesystem administration.

There is no general Permission System, human approval engine, Tool Registry,
subprocess execution, or network sandbox in Phase 0. No security claim is made
for Windows; unsupported platforms fail explicitly.
