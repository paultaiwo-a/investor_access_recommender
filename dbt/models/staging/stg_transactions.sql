-- One row per investment transaction. Buys are the positive implicit feedback.
select
    cast(transactionID as bigint)            as transaction_id,
    customerID                               as customer_id,
    ISIN                                     as isin,
    lower(transactionType)                   as transaction_type,
    cast("timestamp" as date)                as transaction_date,
    cast(totalValue as double)               as total_value,
    cast(units as double)                    as units,
    channel,
    marketID                                 as market_id
from {{ source('far_trans', 'transactions') }}
where customerID is not null and ISIN is not null
-- A handful of transaction ids repeat; keep one row each.
qualify row_number() over (partition by transactionID order by "timestamp") = 1
