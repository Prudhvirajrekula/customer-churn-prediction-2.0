DROP VIEW IF EXISTS rfm_features;

CREATE VIEW rfm_features AS
SELECT
    customer_id,
    last_interaction AS recency,
    usage_frequency AS monthly_avg,
    total_spend AS monetary
FROM raw_customers;
