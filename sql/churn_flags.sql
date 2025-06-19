DROP VIEW IF EXISTS churn_flags;

CREATE VIEW churn_flags AS
SELECT
    customer_id,
    churn AS is_churned
FROM raw_customers;
