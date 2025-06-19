DROP VIEW IF EXISTS model_features;

CREATE VIEW model_features AS
SELECT
    rf.customer_id,
    cf.is_churned,
    rf.recency,
    rf.monthly_avg,
    rf.monetary,
    pb.support_calls,
    pb.payment_delay
FROM rfm_features rf
JOIN churn_flags cf ON rf.customer_id = cf.customer_id
JOIN payment_behavior pb ON rf.customer_id = pb.customer_id;
