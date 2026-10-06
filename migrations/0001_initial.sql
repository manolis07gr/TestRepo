-- 0001: core metadata, mapping registry, raw-capture index, experiments, accounting.
-- Portable SQL (SQLite + PostgreSQL). Decimals are stored as TEXT to stay exact.

CREATE TABLE IF NOT EXISTS contracts (
    venue TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    native_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    series_id TEXT,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    open_ts_ns BIGINT,
    close_ts_ns BIGINT,
    resolve_ts_ns BIGINT,
    tick_size TEXT NOT NULL,
    contract_json TEXT NOT NULL,
    first_seen_ns BIGINT NOT NULL,
    updated_ns BIGINT NOT NULL,
    PRIMARY KEY (venue, contract_id)
);

CREATE TABLE IF NOT EXISTS contract_mappings (
    venue TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    review_status TEXT NOT NULL,
    reviewer TEXT,
    reviewed_at_ns BIGINT,
    mapping_json TEXT NOT NULL,
    created_at_ns BIGINT NOT NULL,
    PRIMARY KEY (venue, contract_id, version)
);

CREATE TABLE IF NOT EXISTS raw_partitions (
    partition_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    stream TEXT NOT NULL,
    day TEXT NOT NULL,
    path TEXT NOT NULL,
    n_records BIGINT NOT NULL,
    sha256 TEXT,
    sealed INTEGER NOT NULL DEFAULT 0,
    created_at_ns BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_event_keys (
    dedup_key TEXT PRIMARY KEY,
    partition_id TEXT NOT NULL,
    recv_ts_ns BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_incidents (
    incident_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    instrument_id TEXT,
    kind TEXT NOT NULL,
    start_ns BIGINT NOT NULL,
    end_ns BIGINT,
    detail TEXT NOT NULL,
    created_at_ns BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    created_at_ns BIGINT NOT NULL,
    git_commit TEXT,
    config_hash TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    decision TEXT,
    report_path TEXT,
    manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypothesis_tests (
    test_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    statistic DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    n_obs BIGINT,
    detail_json TEXT NOT NULL,
    created_at_ns BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS final_test_access (
    access_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    accessed_at_ns BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_fills (
    run_id TEXT NOT NULL,
    fill_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    side TEXT NOT NULL,
    fill_ts_ns BIGINT NOT NULL,
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    fee TEXT NOT NULL,
    fee_schedule_version TEXT NOT NULL,
    liquidity_role TEXT NOT NULL,
    PRIMARY KEY (run_id, fill_id)
);

CREATE TABLE IF NOT EXISTS paper_state (
    run_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at_ns BIGINT NOT NULL,
    PRIMARY KEY (run_id, state_key)
);
