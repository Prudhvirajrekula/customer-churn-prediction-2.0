DROP VIEW IF EXISTS payment_behavior;

CREATE VIEW payment_behavior AS
SELECT
    customer_id,
    support_calls,
    payment_delay
FROM raw_customers;
