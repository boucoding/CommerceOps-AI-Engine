-- Corpus statistics for BM25 scoring of full-text candidates (see src/retrieval/bm25.py).
-- Single row, refreshed whenever the catalog is synchronized.
CREATE TABLE lexical_stats (
    id         BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    stats      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
