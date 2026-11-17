-- Core (cleaned, conformed) tables and ETL control tables.
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS etl;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS core.customer (
    customer_id   VARCHAR PRIMARY KEY,
    first_name    VARCHAR,
    last_name     VARCHAR,
    email         VARCHAR,            -- NULL when the source value was invalid
    phone_e164    VARCHAR,
    country_code  VARCHAR NOT NULL,   -- IE / GB / NI
    county        VARCHAR,
    signup_date   DATE NOT NULL,
    last_batch_id VARCHAR NOT NULL    -- SCD type 1: latest batch wins
);
CREATE TABLE IF NOT EXISTS core.product (
    product_id     VARCHAR PRIMARY KEY,
    product_name   VARCHAR NOT NULL,
    category       VARCHAR NOT NULL,
    list_price_eur DECIMAL(12, 2) NOT NULL,
    unit_cost_eur  DECIMAL(12, 2),
    last_batch_id  VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS core.fx_rate (
    month        VARCHAR,
    currency     VARCHAR,
    eur_per_unit DECIMAL(10, 4) NOT NULL,
    PRIMARY KEY (month, currency)
);
CREATE TABLE IF NOT EXISTS core.sales_order (
    order_id      VARCHAR PRIMARY KEY,
    customer_id   VARCHAR NOT NULL,
    order_ts      TIMESTAMP NOT NULL,
    channel       VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    currency      VARCHAR NOT NULL,
    last_batch_id VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS core.order_line (
    order_id      VARCHAR,
    line_no       INTEGER,
    product_id    VARCHAR NOT NULL,
    quantity      INTEGER NOT NULL,
    unit_price    DECIMAL(12, 2) NOT NULL,   -- in the order currency
    discount_pct  DECIMAL(4, 2) NOT NULL,
    is_return     BOOLEAN NOT NULL,
    last_batch_id VARCHAR NOT NULL,
    PRIMARY KEY (order_id, line_no)
);
CREATE TABLE IF NOT EXISTS etl.batch_log (
    batch_id   VARCHAR,
    file_name  VARCHAR,
    sha256     VARCHAR,
    run_id     INTEGER,
    loaded_at  TIMESTAMP,
    PRIMARY KEY (batch_id, file_name, run_id)
);
CREATE TABLE IF NOT EXISTS etl.quarantine (
    table_name     VARCHAR,
    record_key     VARCHAR,
    reason         VARCHAR,
    payload        VARCHAR,
    batch_id       VARCHAR,
    resolved_batch VARCHAR,           -- set when a retry succeeds (late-arriving parent)
    PRIMARY KEY (table_name, record_key, reason)
);
CREATE TABLE IF NOT EXISTS etl.run_stats (
    run_id     INTEGER,
    batch_id   VARCHAR,
    table_name VARCHAR,
    metric     VARCHAR,
    value      BIGINT,
    PRIMARY KEY (run_id, batch_id, table_name, metric)
);
CREATE TABLE IF NOT EXISTS etl.dq_result (
    run_id    INTEGER,
    check_name VARCHAR,
    severity  VARCHAR,     -- error = stop the run, warn = report only
    passed    BOOLEAN,
    observed  DOUBLE,
    detail    VARCHAR,
    PRIMARY KEY (run_id, check_name)
);
