"""Thin metadata-store wrapper over DB-API connections with versioned migrations.

SQLite is the zero-dependency default (tests, single-workstation research). PostgreSQL
is supported through psycopg (``postgresql://...`` URLs, optional extra ``[postgres]``).
SQL is written with ``?`` placeholders and translated for PostgreSQL.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            path = url.removeprefix("sqlite:///")
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn: Any = sqlite3.connect(path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
        elif url.startswith(("postgresql://", "postgres://")):
            try:
                import psycopg  # type: ignore[import-not-found,unused-ignore]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install the [postgres] extra for PostgreSQL") from exc
            self.dialect = "postgresql"  # pragma: no cover
            self._conn = psycopg.connect(url, autocommit=True)  # pragma: no cover
        else:
            raise ValueError(f"unsupported database url {url!r}")
        self._in_tx = False

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgresql" else sql

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        cur = self._conn.execute(self._sql(sql), tuple(params))
        return int(cur.rowcount or 0)

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if self.dialect == "sqlite":
            self._conn.executemany(sql, [tuple(r) for r in rows])
        else:  # pragma: no cover
            with self._conn.cursor() as cur:
                cur.executemany(self._sql(sql), [tuple(r) for r in rows])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._conn.execute(self._sql(sql), tuple(params))
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        rows = self.query(sql, params)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    @contextmanager
    def transaction(self) -> Iterator[Database]:
        if self._in_tx:  # nested: join the outer transaction
            yield self
            return
        self._conn.execute("BEGIN")
        self._in_tx = True
        try:
            yield self
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._in_tx = False

    def close(self) -> None:
        self._conn.close()

    def table_names(self) -> set[str]:
        if self.dialect == "sqlite":
            rows = self.query("SELECT name FROM sqlite_master WHERE type='table'")
        else:  # pragma: no cover
            rows = self.query(
                "SELECT table_name AS name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        return {str(r["name"]) for r in rows}


def _split_statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def migrate(db: Database, migrations_dir: Path | None = None) -> list[str]:
    """Apply pending ``NNNN_name.sql`` migrations in order; returns the applied names."""
    directory = migrations_dir or DEFAULT_MIGRATIONS_DIR
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version TEXT PRIMARY KEY, applied_at_ns BIGINT NOT NULL)"
    )
    applied = {str(r["version"]) for r in db.query("SELECT version FROM schema_migrations")}
    done: list[str] = []
    for path in sorted(directory.glob("*.sql")):
        if path.stem in applied:
            continue
        with db.transaction():
            for stmt in _split_statements(path.read_text(encoding="utf-8")):
                db.execute(stmt)
            db.execute(
                "INSERT INTO schema_migrations (version, applied_at_ns) VALUES (?, ?)",
                (path.stem, time.time_ns()),
            )
        done.append(path.stem)
    return done


def open_database(url: str, *, run_migrations: bool = True) -> Database:
    db = Database(url)
    if run_migrations:
        migrate(db)
    return db
