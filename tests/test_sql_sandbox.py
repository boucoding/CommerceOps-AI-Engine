"""
SQL sandbox: static validation (no database needed) and database-enforced limits.
"""

import pytest

from src.sandbox import sql as sandbox_sql
from src.sandbox.sql import SQLRejected, validate_sql


@pytest.mark.parametrize(
    "query",
    [
        "SELECT name, price FROM product_catalog WHERE stock > 0 ORDER BY price DESC",
        "select product_id, sum(quantity) as units from order_history where status = 'created' "
        "group by product_id order by units desc limit 3",
        "WITH best AS (SELECT product_id, count(*) AS n FROM order_history GROUP BY product_id) "
        "SELECT c.name, b.n FROM best b JOIN product_catalog c USING (product_id)",
        "SELECT category, round(avg(price), 2) FROM product_catalog GROUP BY category;",
        "SELECT name FROM product_catalog WHERE lower(name) LIKE '%beanie%' UNION SELECT 'x'",
        "SELECT name, rank() OVER (PARTITION BY category ORDER BY price) FROM product_catalog",
    ],
)
def test_accepts_read_only_analytics(query):
    safe = validate_sql(query, max_rows=50)
    assert safe.startswith("SELECT * FROM (") and safe.endswith("LIMIT 51")


@pytest.mark.parametrize(
    "query, reason",
    [
        ("UPDATE products SET stock = 0", "only SELECT"),
        ("DELETE FROM order_history", "only SELECT"),
        ("DROP TABLE products", "only SELECT"),
        ("SELECT 1; DROP TABLE products", "exactly one statement"),
        ("SELECT * FROM emails", "forbidden relation"),
        ("SELECT * FROM products", "forbidden relation"),
        ("SELECT * FROM public.product_catalog", "schema-qualified"),
        ("SELECT * FROM pg_catalog.pg_roles", "schema-qualified"),
        ("SELECT * FROM pg_user", "forbidden relation"),
        ("SELECT * FROM information_schema.tables", "schema-qualified"),
        ("SELECT set_config('role', 'postgres', true)", "function SET_CONFIG"),
        ("SELECT pg_sleep(10)", "function PG_SLEEP"),
        ("SELECT pg_read_file('/etc/passwd')", "function PG_READ_FILE"),
        ("SELECT * FROM dblink('host=evil', 'select 1') AS t(x int)", "not allowed"),
        ("SELECT current_setting('is_superuser')", "function CURRENT_SETTING"),
        ("SELECT * INTO stolen FROM product_catalog", "not allowed"),
        ("SELECT * FROM product_catalog FOR UPDATE", "not allowed"),
        ("WITH x AS (DELETE FROM order_history RETURNING *) SELECT * FROM x", "not allowed"),
        ("COPY product_catalog TO '/tmp/x'", "only SELECT"),
        ("", "empty"),
        ("SELEKT name FROM product_catalog", "parse"),
    ],
)
def test_rejects_unsafe_sql(query, reason):
    with pytest.raises(SQLRejected, match=reason):
        validate_sql(query, max_rows=50)


# ------------------------------------------------------------------ database layer


def test_results_are_capped_and_audited(make_engine):
    engine = make_engine(SQL_MAX_ROWS=3, SQL_MAX_CELL_CHARS=12)
    result = engine.sandbox.run(
        "SELECT product_id, description FROM product_catalog ORDER BY product_id", email_id="T1", purpose="test"
    )
    assert len(result.rows) == 3 and result.truncated
    assert all(len(row[1]) <= 12 for row in result.rows)
    with engine.store.db.cursor() as cur:
        cur.execute("SELECT outcome, row_count FROM sql_audit WHERE email_id = 'T1'")
        assert cur.fetchall() == [("ok", 3)]


def test_rejections_are_audited(make_engine):
    engine = make_engine()
    with pytest.raises(SQLRejected):
        engine.sandbox.run("DELETE FROM order_history", email_id="T2", purpose="attack")
    with engine.store.db.cursor() as cur:
        cur.execute("SELECT outcome FROM sql_audit WHERE email_id = 'T2'")
        assert cur.fetchall() == [("rejected",)]


@pytest.fixture
def unvalidated(monkeypatch):
    """Bypass the parser to prove the database itself enforces the boundary."""
    monkeypatch.setattr(sandbox_sql, "validate_sql", lambda raw, max_rows: raw)


@pytest.mark.parametrize(
    "query, error",
    [
        ("UPDATE products SET stock = 0", "read-only transaction"),
        ("SELECT email_id FROM emails", "permission denied"),
        ("SELECT product_id FROM products", "permission denied"),
        ("SELECT * FROM sql_audit", "permission denied"),
    ],
)
def test_database_enforces_read_only_role(make_engine, unvalidated, query, error):
    engine = make_engine()
    with pytest.raises(SQLRejected, match=error):
        engine.sandbox.run(query, email_id=None, purpose="bypass test")
    assert engine.store.get_products(["SWL2345"])["SWL2345"].stock == 5


def test_statement_timeout(make_engine, unvalidated):
    engine = make_engine(SQL_STATEMENT_TIMEOUT_MS=100)
    with pytest.raises(SQLRejected, match="time limit"):
        engine.sandbox.run("SELECT pg_sleep(2)", email_id=None, purpose="slow")
