"""POSIX workspace text I/O using descriptor-relative, no-follow traversal.

This is a file capability boundary, not an OS sandbox. A caller that can run
arbitrary Python or shell commands can bypass it. See docs/security.md.
"""

import errno
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from types import TracebackType


class FileErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    PATH_DENIED = "path_denied"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    NOT_FILE = "not_file"
    ALREADY_EXISTS = "already_exists"
    TOO_LARGE = "too_large"
    INVALID_TEXT = "invalid_text"
    IO_ERROR = "io_error"
    CLOSED = "closed"


class FileToolError(Exception):
    """Stable error code and safe message, without disclosing host paths."""

    def __init__(self, code: FileErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


def _os_error(exc: OSError) -> FileToolError:
    mapping = {
        errno.ENOENT: (FileErrorCode.NOT_FOUND, "file or parent directory does not exist"),
        errno.EACCES: (FileErrorCode.PERMISSION_DENIED, "operating system denied access"),
        errno.EPERM: (FileErrorCode.PERMISSION_DENIED, "operating system denied access"),
        errno.ELOOP: (FileErrorCode.PATH_DENIED, "symbolic links are not allowed"),
        errno.ENOTDIR: (FileErrorCode.NOT_FILE, "a parent component is not a directory"),
        errno.EISDIR: (FileErrorCode.NOT_FILE, "expected a regular file"),
        errno.EEXIST: (FileErrorCode.ALREADY_EXISTS, "destination already exists"),
    }
    number = exc.errno if exc.errno is not None else 0
    code, message = mapping.get(number, (FileErrorCode.IO_ERROR, "file operation failed"))
    return FileToolError(code, message)


class WorkspaceFiles:
    """Read UTF-8 files; explicitly opt in to atomic create/replace operations.

    Parents must already exist. Paths must be canonical, forward-slash relative
    strings. Use as a context manager to close the pinned workspace descriptor.
    Every symlink below the trusted root is rejected, even links targeting it.
    """

    def __init__(
        self, root: str | Path, *, allow_write: bool = False, max_bytes: int = 1_048_576
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("workspace file tools require POSIX (macOS or Linux)")
        if not isinstance(root, (str, Path)) or not str(root) or "\x00" in str(root):
            raise FileToolError(FileErrorCode.INVALID_INPUT, "root must be a nonempty path")
        if type(allow_write) is not bool or type(max_bytes) is not int or max_bytes <= 0:
            raise FileToolError(FileErrorCode.INVALID_INPUT, "invalid workspace options")
        self._allow_write = allow_write
        self._max_bytes = max_bytes
        try:
            self._root = Path(root).resolve(strict=True)
            self._root_fd = os.open(self._root, self._directory_flags)
        except OSError as exc:
            raise _os_error(exc) from None
        except RuntimeError:
            raise FileToolError(
                FileErrorCode.PATH_DENIED, "root contains a symbolic link loop"
            ) from None

    @property
    def _directory_flags(self) -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    @property
    def root(self) -> Path:
        return self._root

    def close(self) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def __enter__(self) -> "WorkspaceFiles":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._root_fd < 0:
            raise FileToolError(FileErrorCode.CLOSED, "workspace is closed")

    def _parts(self, path: str) -> list[str]:
        if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
            raise FileToolError(FileErrorCode.INVALID_INPUT, "expected a relative file path")
        parts = path.split("/")
        if path.startswith("/") or any(part == ".." for part in parts):
            raise FileToolError(FileErrorCode.PATH_DENIED, "path must stay inside the workspace")
        if any(part in ("", ".") for part in parts):
            raise FileToolError(FileErrorCode.INVALID_INPUT, "path must be canonical and nonempty")
        if any(
            part.casefold() in (".git", ".venv", ".env") or part.casefold().startswith(".env.")
            for part in parts
        ):
            raise FileToolError(FileErrorCode.PATH_DENIED, "protected path is not accessible")
        return parts

    @contextmanager
    def _parent(self, path: str) -> Iterator[tuple[int, str]]:
        self._ensure_open()
        parts = self._parts(path)
        fd = -1
        try:
            fd = os.dup(self._root_fd)
            for part in parts[:-1]:
                if stat.S_ISLNK(os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode):
                    raise FileToolError(FileErrorCode.PATH_DENIED, "symbolic links are not allowed")
                child = os.open(part, self._directory_flags, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        except OSError as exc:
            raise _os_error(exc) from None
        finally:
            if fd >= 0:
                os.close(fd)

    def _check_regular(self, info: os.stat_result) -> None:
        if stat.S_ISLNK(info.st_mode):
            raise FileToolError(FileErrorCode.PATH_DENIED, "symbolic links are not allowed")
        if not stat.S_ISREG(info.st_mode):
            raise FileToolError(FileErrorCode.NOT_FILE, "expected a regular file")
        if info.st_nlink != 1:
            raise FileToolError(FileErrorCode.PATH_DENIED, "multiply linked files are not allowed")

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
            raise FileToolError(FileErrorCode.INVALID_INPUT, "invalid read byte limit")
        limit = self._max_bytes if max_bytes is None else min(self._max_bytes, max_bytes)
        with self._parent(path) as (parent, name):
            self._check_regular(os.stat(name, dir_fd=parent, follow_symlinks=False))
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            fd = os.open(name, flags, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                self._check_regular(os.fstat(stream.fileno()))
                if os.fstat(stream.fileno()).st_size > limit:
                    raise FileToolError(FileErrorCode.TOO_LARGE, "file exceeds byte limit")
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise FileToolError(FileErrorCode.TOO_LARGE, "file exceeds byte limit")
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                raise FileToolError(FileErrorCode.INVALID_TEXT, "file is not valid UTF-8") from None

    def list_directories(self, path: str, *, max_entries: int = 128) -> tuple[str, ...]:
        """Bounded, descriptor-relative discovery; reject every symlink entry."""
        if type(max_entries) is not int or not 1 <= max_entries <= 1024:
            raise FileToolError(FileErrorCode.INVALID_INPUT, "invalid listing limit")
        self._ensure_open()
        self._parts(path)
        with self._parent(path + "/__entries__") as (directory, _):
            names: list[str] = []
            with os.scandir(directory) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > max_entries:
                        raise FileToolError(
                            FileErrorCode.TOO_LARGE, "directory exceeds entry limit"
                        )
                    if entry.is_symlink():
                        raise FileToolError(
                            FileErrorCode.PATH_DENIED, "symbolic links are not allowed"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        self._parts(path + "/" + entry.name)
                        names.append(entry.name)
            return tuple(sorted(names))

    def read_frontmatter(self, path: str, *, max_bytes: int = 4096) -> str:
        """Read only the opening --- block, without buffering or reading the body."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise FileToolError(FileErrorCode.INVALID_INPUT, "invalid header byte limit")
        limit = min(self._max_bytes, max_bytes)
        with self._parent(path) as (parent, name):
            self._check_regular(os.stat(name, dir_fd=parent, follow_symlinks=False))
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            fd = os.open(name, flags, dir_fd=parent)
            # FileIO.readline is unbuffered: closing --- never pulls body bytes ahead.
            with os.fdopen(fd, "rb", buffering=0) as stream:
                self._check_regular(os.fstat(stream.fileno()))
                data = bytearray()
                while True:
                    line = stream.readline(limit - len(data) + 1)
                    first = not data
                    data.extend(line)
                    if len(data) > limit:
                        raise FileToolError(FileErrorCode.TOO_LARGE, "header exceeds byte limit")
                    if not line or (first and line.rstrip(b"\r\n") != b"---"):
                        raise FileToolError(FileErrorCode.INVALID_TEXT, "missing frontmatter")
                    if len(data) > len(line) and line.rstrip(b"\r\n") == b"---":
                        break
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    raise FileToolError(FileErrorCode.INVALID_TEXT, "header is not UTF-8") from None

    def write_text(self, path: str, content: str, *, overwrite: bool = False) -> None:
        self._ensure_open()
        if not self._allow_write:
            raise FileToolError(FileErrorCode.PERMISSION_DENIED, "workspace writes are disabled")
        if not isinstance(content, str) or type(overwrite) is not bool:
            raise FileToolError(FileErrorCode.INVALID_INPUT, "invalid content or overwrite option")
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            raise FileToolError(FileErrorCode.INVALID_TEXT, "content is not valid UTF-8") from None
        if len(data) > self._max_bytes:
            raise FileToolError(FileErrorCode.TOO_LARGE, "content exceeds byte limit")
        with self._parent(path) as (parent, name):
            try:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                self._check_regular(info)
                if not overwrite:
                    raise FileToolError(FileErrorCode.ALREADY_EXISTS, "destination already exists")
            temporary = ".harness-" + uuid.uuid4().hex + ".tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(temporary, flags, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if overwrite:
                    os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                else:
                    os.link(
                        temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False
                    )
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
