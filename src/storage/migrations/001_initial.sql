-- CommerceOps AI Engine schema: catalog, emails and order lines, plus the agent's read-only SQL sandbox.
-- Applied by src/storage/migrate.py inside a transaction; recorded in schema_migrations.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Product catalog with semantic embeddings and a full-text search column.
CREATE TABLE products (
    product_id      VARCHAR(32) PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    stock           INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    seasons         TEXT NOT NULL DEFAULT '',
    price           NUMERIC(10, 2) NOT NULL DEFAULT 0 CHECK (price >= 0),
    embedding       vector(1536),
    embedding_model TEXT,
    embedding_hash  TEXT,
    search_tsv      tsvector GENERATED ALWAYS AS (
        to_tsvector('english', name || ' ' || category || ' ' || seasons || ' ' || description)
    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for approximate nearest-neighbour search on cosine distance.
CREATE INDEX products_embedding_hnsw_idx ON products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- Trigram index for typo-tolerant product-ID lookup; full-text index for keyword search.
CREATE INDEX products_id_trgm_idx ON products USING gin (product_id gin_trgm_ops);
CREATE INDEX products_search_tsv_idx ON products USING gin (search_tsv);
CREATE INDEX products_category_idx ON products (category);

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER products_set_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- One row per processed email. Raw bodies are intentionally not stored (data minimization);
-- `outcome` holds the auditable processing result.
CREATE TABLE emails (
    email_id     VARCHAR(64) PRIMARY KEY,
    category     TEXT,
    status       TEXT NOT NULL CHECK (status IN ('processing', 'processed', 'needs_clarification',
                                                 'escalated', 'needs_review', 'failed')),
    outcome      JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ
);
CREATE INDEX emails_status_idx ON emails (status);

CREATE TABLE order_lines (
    line_id            BIGSERIAL PRIMARY KEY,
    email_id           VARCHAR(64) NOT NULL REFERENCES emails (email_id) ON DELETE CASCADE,
    line_no            INTEGER NOT NULL,
    product_id         VARCHAR(32) NOT NULL REFERENCES products (product_id),
    product_name       TEXT NOT NULL,
    requested_quantity INTEGER NOT NULL CHECK (requested_quantity > 0),
    fulfilled_quantity INTEGER NOT NULL CHECK (fulfilled_quantity >= 0),
    status             TEXT NOT NULL CHECK (status IN ('created', 'out of stock')),
    remaining_stock    INTEGER NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (email_id, line_no),
    UNIQUE (email_id, product_id)  -- a retried email never reserves the same product twice
);
CREATE INDEX order_lines_product_idx ON order_lines (product_id);

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
