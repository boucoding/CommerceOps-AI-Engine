-- Agent support: read-only SQL sandbox, SQL audit log, per-product reservation idempotency.

-- BM25 statistics were only needed for in-memory/PostgreSQL ranking parity, which is gone.
DROP TABLE IF EXISTS lexical_stats;

-- Fallback/degraded processing no longer exists.
ALTER TABLE emails DROP COLUMN degraded;

-- A retried email must never reserve the same product twice.
ALTER TABLE order_lines ADD CONSTRAINT order_lines_email_product_key UNIQUE (email_id, product_id);

-- The only relations the agent's SQL may read. No customer data (emails, replies) is exposed.
CREATE VIEW product_catalog AS
    SELECT product_id, name, category, description, seasons, price, stock FROM products;

CREATE VIEW order_history AS
    SELECT product_id, product_name, requested_quantity AS quantity, status, created_at AS ordered_at
    FROM order_lines;

-- NOLOGIN role assumed with SET LOCAL ROLE inside a READ ONLY transaction.
-- Role names are cluster-wide, so creation is idempotent across databases.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'commerceops_agent_readonly') THEN
        CREATE ROLE commerceops_agent_readonly NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;
GRANT USAGE ON SCHEMA public TO commerceops_agent_readonly;
GRANT SELECT ON product_catalog, order_history TO commerceops_agent_readonly;
GRANT commerceops_agent_readonly TO CURRENT_USER;

CREATE TABLE sql_audit (
    audit_id   BIGSERIAL PRIMARY KEY,
    email_id   VARCHAR(64),
    purpose    TEXT NOT NULL,
    sql_text   TEXT NOT NULL,
    outcome    TEXT NOT NULL CHECK (outcome IN ('ok', 'rejected', 'error')),
    error      TEXT,
    row_count  INTEGER NOT NULL DEFAULT 0,
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX sql_audit_email_idx ON sql_audit (email_id);
