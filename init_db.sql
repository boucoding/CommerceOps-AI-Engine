-- ==============================================================================
-- CommerceOps AI Engine - PostgreSQL Schema Initialization
-- Includes: pgvector (HNSW Indexing) and pg_trgm (Fuzzy Trigram Matching)
-- ==============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Products Catalog Table with Vector Embeddings
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    stock INT NOT NULL DEFAULT 0 CHECK (stock >= 0),
    seasons VARCHAR(100),
    price NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. High-Performance HNSW Vector Index (Sub-millisecond Cosine Similarity Search)
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx 
ON products USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Trigram Fuzzy Indexes for Instant SKU & Name Typo Recovery
CREATE INDEX IF NOT EXISTS products_id_trgm_idx ON products USING gin (product_id gin_trgm_ops);
CREATE INDEX IF NOT EXISTS products_name_trgm_idx ON products USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);

-- 5. Customer Inquiries & Communications Audit Log
CREATE TABLE IF NOT EXISTS customer_emails (
    email_id VARCHAR(50) PRIMARY KEY,
    subject TEXT,
    body TEXT NOT NULL,
    intent_category VARCHAR(50),
    intent_confidence NUMERIC(4, 3),
    raw_response TEXT,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Orders Ledger
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    email_id VARCHAR(50) REFERENCES customer_emails(email_id),
    status VARCHAR(50) NOT NULL DEFAULT 'processing', -- 'created', 'partially_fulfilled', 'out_of_stock'
    total_items INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Order Lines with Status
CREATE TABLE IF NOT EXISTS order_lines (
    line_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id) ON DELETE CASCADE,
    email_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    requested_quantity INT NOT NULL CHECK (requested_quantity > 0),
    fulfilled_quantity INT NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL, -- 'created', 'out of stock'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. Trigger to Update updated_at Timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_products_updated_at
BEFORE UPDATE ON products
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
