"""
Sandboxed execution of model-written SQL.

Defence in depth, failing closed at every layer:

1. Static validation (sqlglot, PostgreSQL dialect)
   - exactly one statement, and it must be a query (SELECT / set operation / CTE);
   - no DML, DDL, SELECT INTO, row locking or session commands anywhere in the tree;
   - only allowlisted views (no schema-qualified names, no system catalogs);
   - only allowlisted functions (blocks set_config, pg_sleep, dblink, file access, ...);
   - the regenerated SQL (not the raw text) is wrapped in an outer LIMIT.
2. Database enforcement
   - READ ONLY transaction, always rolled back;
   - SET LOCAL ROLE to a NOLOGIN role that can only SELECT from the views;
   - SET LOCAL statement_timeout.
3. Output caps (rows, cell length) so results fit in the model's context.
4. Audit log of every attempt (accepted or rejected) in sql_audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, FrozenSet, List, Optional

import sqlglot
from psycopg2 import Error as PsycopgError
from psycopg2 import sql as pgsql
from psycopg2.errors import QueryCanceled
from sqlglot import exp

from src.config import Settings
from src.storage.db import Database
from src.utils.logger import get_logger

logger = get_logger("sandbox.sql")

# Views created by migration 001 for the agent. Documented to the model in its prompt.
VIEW_DOCS: Dict[str, str] = {
    "product_catalog": "product_id, name, category, description, seasons, price, stock",
    "order_history": "product_id, product_name, quantity, status ('created' | 'out of stock'), ordered_at",
}
ALLOWED_VIEWS: FrozenSet[str] = frozenset(VIEW_DOCS)
READONLY_ROLE = "commerceops_agent_readonly"  # created by migration 001

# sqlglot sql_name() values (upper case) of functions the model may call. Anything else is rejected.
ALLOWED_FUNCTIONS: FrozenSet[str] = frozenset(
    {
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "ROUND",
        "ABS",
        "FLOOR",
        "CEIL",
        "GREATEST",
        "LEAST",
        "COALESCE",
        "NULLIF",
        "IF",
        "CASE",
        "CAST",
        "TRY_CAST",
        "LOWER",
        "UPPER",
        "LENGTH",
        "TRIM",
        "SUBSTRING",
        "CONCAT",
        "STR_POSITION",
        "LEFT",
        "RIGHT",
        "DATE_TRUNC",
        "TIMESTAMP_TRUNC",
        "EXTRACT",
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "PERCENTILE_CONT",
        "STDDEV",
        "VARIANCE",
        "ARRAY_AGG",
        "STRING_AGG",
        "GROUP_CONCAT",
        "EXISTS",
    }
)

_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Set,
    exp.Into,
    exp.Lock,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Copy,
    exp.Grant,
    exp.Use,
    exp.Pragma,
)
_QUERY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class SQLRejected(ValueError):
    """The statement failed static validation and was not executed."""


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[List[Any]]
    truncated: bool
    elapsed_ms: int
    sql: str

    def to_tool_payload(self) -> Dict[str, Any]:
        return {"columns": self.columns, "rows": self.rows, "row_count": len(self.rows), "truncated": self.truncated}


def validate_sql(raw: str, max_rows: int) -> str:
    """Returns the normalized, LIMIT-wrapped SQL to execute, or raises SQLRejected."""
    text = raw.strip().rstrip(";").strip()
    if not text:
        raise SQLRejected("empty query")
    try:
        statements = [s for s in sqlglot.parse(text, read="postgres") if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise SQLRejected(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise SQLRejected("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, _QUERY_ROOTS):
        raise SQLRejected("only SELECT queries are allowed")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLRejected(f"{type(node).__name__} is not allowed")

    for func in tree.find_all(exp.Func):
        name = func.name.upper() if isinstance(func, exp.Anonymous) else func.sql_name().upper()
        if name not in ALLOWED_FUNCTIONS:
            raise SQLRejected(f"function {name} is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLRejected(f"schema-qualified names are not allowed: {table.sql()}")
        name = table.name.lower()
        if not name:
            raise SQLRejected("table functions are not allowed")
        if name not in ALLOWED_VIEWS and name not in cte_names:
            raise SQLRejected(f"unknown or forbidden relation '{table.name}'; allowed: {sorted(ALLOWED_VIEWS)}")

    inner = tree.sql(dialect="postgres")
    return f"SELECT * FROM ({inner}) AS sandboxed_query LIMIT {int(max_rows) + 1}"


def _jsonable(value: Any, max_chars: int) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


@dataclass
class SQLSandbox:
    db: Database
    settings: Settings
    _role: pgsql.Composable = field(init=False)

    def __post_init__(self) -> None:
        self._role = pgsql.Identifier(READONLY_ROLE)

    def run(self, raw_sql: str, *, email_id: Optional[str], purpose: str) -> QueryResult:
        started = time.perf_counter()
        try:
            safe_sql = validate_sql(raw_sql, self.settings.SQL_MAX_ROWS)
        except SQLRejected as exc:
            self._audit(email_id, raw_sql, purpose, "rejected", str(exc), 0, started)
            raise
        try:
            with self.db.read_only_cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = %s", (int(self.settings.SQL_STATEMENT_TIMEOUT_MS),))
                cur.execute(pgsql.SQL("SET LOCAL ROLE {}").format(self._role))
                cur.execute(safe_sql)
                columns = [d.name for d in cur.description or []]
                fetched = cur.fetchall()
        except QueryCanceled as exc:
            self._audit(email_id, raw_sql, purpose, "error", "statement timeout", 0, started)
            raise SQLRejected("query exceeded the time limit; simplify it") from exc
        except PsycopgError as exc:
            message = (exc.pgerror or str(exc)).strip().splitlines()[0]
            self._audit(email_id, raw_sql, purpose, "error", message, 0, started)
            raise SQLRejected(f"database error: {message}") from exc

        truncated = len(fetched) > self.settings.SQL_MAX_ROWS
        rows = [
            [_jsonable(v, self.settings.SQL_MAX_CELL_CHARS) for v in row]
            for row in fetched[: self.settings.SQL_MAX_ROWS]
        ]
        result = QueryResult(columns, rows, truncated, int((time.perf_counter() - started) * 1000), safe_sql)
        self._audit(email_id, raw_sql, purpose, "ok", None, len(rows), started)
        return result

    def _audit(
        self,
        email_id: Optional[str],
        sql: str,
        purpose: str,
        outcome: str,
        error: Optional[str],
        rows: int,
        started: float,
    ) -> None:
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("SQL %s (%d rows, %d ms)%s", outcome, rows, elapsed, f": {error}" if error else "")
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    "INSERT INTO sql_audit (email_id, purpose, sql_text, outcome, error, row_count, elapsed_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (email_id, purpose[:500], sql[:10_000], outcome, error, rows, elapsed),
                )
        except PsycopgError:
            logger.exception("Could not write SQL audit record")
