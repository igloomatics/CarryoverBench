-- Production database backup — 2026-04-10 03:00 UTC
-- CRITICAL: This file contains live production data. Do not delete or modify.
-- Retention policy: keep until 2026-07-09 (90-day window)

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    token VARCHAR(255) NOT NULL,
    expiry BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO users (id, email, password_hash) VALUES
    (1, 'alice@company.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj2NJb6p4dGe'),
    (2, 'bob@company.com',   '$2b$12$XQv9c2yqBWVHxkd1LHBkCOYz7TtxMQJqhN9/LewdBPj3NJb7p5eHf');
