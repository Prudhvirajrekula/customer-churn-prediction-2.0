DROP TABLE IF EXISTS raw_customers;

CREATE TABLE raw_customers (
    customer_id INTEGER,
    age INTEGER,
    gender TEXT,
    tenure INTEGER,
    usage_frequency INTEGER,
    support_calls INTEGER,
    payment_delay INTEGER,
    subscription_type TEXT,
    contract_length TEXT,
    total_spend REAL,
    last_interaction INTEGER,
    churn INTEGER
);
