"""Explicit opt-in SQLite memory for small approved structured facts.

Not conversation history, hidden reasoning, business truth or an authorization
source. Namespaces are logical application scopes, not same-user OS isolation.
"""

import os
import re
import sqlite3
import stat
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_APP_ID = 0x41484D31
_COLUMNS = (
    "namespace",
    "session",
    "id",
    "entity",
    "attribute",
    "value",
    "created_ns",
    "updated_ns",
    "revision",
)
_SCHEMA = """CREATE TABLE records (
namespace TEXT NOT NULL, session TEXT NOT NULL, id TEXT NOT NULL,
entity TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
created_ns INTEGER NOT NULL, updated_ns INTEGER NOT NULL, revision INTEGER NOT NULL,
PRIMARY KEY(namespace, session, id))"""
_LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}\Z")
_SENSITIVE = re.compile(
    r"(?i)(password|credential|secret|api.?key|access.?token|refresh.?token|authorization|reasoning|chain.of.thought|-----BEGIN|\bbearer\b|\bsk-[a-z0-9_-]{8,})"
)


class MemoryFault(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    entity: str
    attribute: str
    value: str
    created_ns: int
    updated_ns: int
    revision: int


def _fields(entity: str, attribute: str, value: str) -> None:
    if (
        type(entity) is not str
        or not _LABEL.fullmatch(entity)
        or type(attribute) is not str
        or not _LABEL.fullmatch(attribute)
        or type(value) is not str
        or not 1 <= len(value.encode()) <= 512
        or not value.isprintable()
        or _SENSITIVE.search(entity + " " + attribute + " " + value)
    ):
        raise MemoryFault("memory_structured_fact_rejected")


def _identifier(value: str) -> None:
    try:
        if type(value) is not str or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise MemoryFault("memory_id_invalid") from None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


class MemoryStore:
    MAX_SCOPE = 64
    MAX_GLOBAL = 256

    def __init__(self, path: Path, namespace: str, session: str, *, enabled: bool = False) -> None:
        if enabled is not True:
            raise MemoryFault("memory_disabled")  # Before any filesystem access.
        if not all(type(s) is str and _LABEL.fullmatch(s) for s in (namespace, session)):
            raise MemoryFault("memory_scope_invalid")
        absolute = Path(os.path.abspath(path))
        if any(p.casefold() == ".git" for p in absolute.parts) or absolute.suffix not in {
            ".db",
            ".sqlite3",
        }:
            raise MemoryFault("memory_path_forbidden")
        for part in reversed((absolute, *absolute.parents)):
            try:
                mode = part.lstat().st_mode
            except FileNotFoundError:
                break
            if stat.S_ISLNK(mode):
                raise MemoryFault("memory_symlink_forbidden")
        absolute.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = absolute.parent.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise MemoryFault("memory_private_directory_required")
        fd = os.open(absolute, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or info.st_size > 2 * 1024 * 1024
            ):
                raise MemoryFault("memory_private_file_required")
        finally:
            os.close(fd)
        self.path, self.namespace, self.session = absolute, namespace, session
        self._validated = False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            application = db.execute("PRAGMA application_id").fetchone()[0]
            objects = db.execute(
                "SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if version == 0 and application == 0 and not objects:
                db.execute(_SCHEMA)
                db.execute("PRAGMA user_version=1")
                db.execute(f"PRAGMA application_id={_APP_ID}")
            elif version != 1 or application != _APP_ID or objects != [("records", "table")]:
                raise MemoryFault("memory_schema_mismatch")
            columns = tuple(row[1] for row in db.execute("PRAGMA table_info(records)"))
            if columns != _COLUMNS:
                raise MemoryFault("memory_schema_mismatch")
            db.commit()
            self._validated = True

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        # The owner-only directory/file is a custody precaution, not protection
        # against arbitrary code under the same macOS/Linux user.
        info = self.path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or info.st_size > 2 * 1024 * 1024
        ):
            raise MemoryFault("memory_private_file_required")
        db = sqlite3.connect(self.path, timeout=2, isolation_level=None)
        try:
            db.execute("PRAGMA trusted_schema=OFF")
            if self._validated:
                db.execute("PRAGMA secure_delete=ON")
                db.execute("PRAGMA max_page_count=512")
            yield db
        except sqlite3.Error:
            if db.in_transaction:
                db.rollback()
            raise MemoryFault("memory_storage_error") from None
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def _record(self, row: tuple[object, ...]) -> MemoryRecord:
        if (
            len(row) != 7
            or not all(type(v) is str for v in row[:4])
            or not all(type(v) is int for v in row[4:])
        ):
            raise MemoryFault("memory_record_invalid")
        record = MemoryRecord(*row)  # type: ignore[arg-type]
        _identifier(record.record_id)
        _fields(record.entity, record.attribute, record.value)
        if not 0 < record.created_ns <= record.updated_ns or record.revision < 1:
            raise MemoryFault("memory_record_invalid")
        return record

    def get(self, record_id: str) -> MemoryRecord | None:
        _identifier(record_id)
        with self._db() as db:
            row = db.execute(
                "SELECT id,entity,attribute,value,created_ns,updated_ns,revision FROM records "
                "WHERE namespace=? AND session=? AND id=?",
                (self.namespace, self.session, record_id),
            ).fetchone()
        return self._record(row) if row else None

    def search(self, query: str, top_k: int = 3) -> tuple[tuple[int, MemoryRecord], ...]:
        if (
            type(query) is not str
            or not 1 <= len(query) <= 256
            or type(top_k) is not int
            or not 1 <= top_k <= 10
        ):
            raise MemoryFault("memory_query_invalid")
        terms = _tokens(query)
        with self._db() as db:
            rows = db.execute(
                "SELECT id,entity,attribute,value,created_ns,updated_ns,revision FROM records "
                "WHERE namespace=? AND session=? ORDER BY id LIMIT 65",
                (self.namespace, self.session),
            ).fetchall()
        if len(rows) > self.MAX_SCOPE:
            raise MemoryFault("memory_capacity_invalid")
        records = [self._record(row) for row in rows]
        ranked = [
            (len(terms & _tokens(r.entity + " " + r.attribute + " " + r.value)), r) for r in records
        ]
        return tuple(
            sorted((r for r in ranked if r[0] > 0), key=lambda r: (-r[0], r[1].record_id))[:top_k]
        )

    def _put(self, entity: str, attribute: str, value: str) -> MemoryRecord:
        """Trusted handler body only; registered mutation must use Core exact ASK."""
        _fields(entity, attribute, value)
        now = time.time_ns()
        record = MemoryRecord(str(uuid.uuid4()), entity, attribute, value, now, now, 1)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT count(*) FROM records").fetchone()[0]
            scoped = db.execute(
                "SELECT count(*) FROM records WHERE namespace=? AND session=?",
                (self.namespace, self.session),
            ).fetchone()[0]
            if total >= self.MAX_GLOBAL or scoped >= self.MAX_SCOPE:
                raise MemoryFault("memory_capacity_reached")
            db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    self.namespace,
                    self.session,
                    record.record_id,
                    entity,
                    attribute,
                    value,
                    now,
                    now,
                    1,
                ),
            )
            db.commit()
        return record

    def _update(self, record_id: str, revision: int, value: str) -> MemoryRecord:
        current = self.get(record_id)
        if current is None or type(revision) is not int or current.revision != revision:
            raise MemoryFault("memory_revision_conflict")
        _fields(current.entity, current.attribute, value)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE records SET value=?,updated_ns=?,revision=revision+1 "
                "WHERE namespace=? AND session=? AND id=? AND revision=?",
                (
                    value,
                    max(time.time_ns(), current.updated_ns + 1),
                    self.namespace,
                    self.session,
                    record_id,
                    revision,
                ),
            ).rowcount
            if changed != 1:
                raise MemoryFault("memory_revision_conflict")
            db.commit()
        result = self.get(record_id)
        if result is None:
            raise MemoryFault("memory_readback_failed")
        return result

    def _delete(self, record_id: str, revision: int) -> None:
        _identifier(record_id)
        if type(revision) is not int or revision < 1:
            raise MemoryFault("memory_revision_conflict")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            deleted = db.execute(
                "DELETE FROM records WHERE namespace=? AND session=? AND id=? AND revision=?",
                (self.namespace, self.session, record_id, revision),
            ).rowcount
            if deleted != 1:
                raise MemoryFault("memory_revision_conflict")
            db.commit()
