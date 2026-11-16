-- Customer x asset buy interactions, split by date:
--   train: [start, val_start)   val: [val_start, test_start)   test: [test_start, test_end)
-- is_new marks the window in which the customer bought the asset for the first time.
-- Evaluation targets are new interactions only, so repeat buys of a held asset never leak
-- across the split.
with buys as (
    select
        customer_id,
        isin,
        transaction_date,
        total_value,
        case
            when transaction_date < date '{{ var("val_start") }}' then 'train'
            when transaction_date < date '{{ var("test_start") }}' then 'val'
            when transaction_date < date '{{ var("test_end") }}' then 'test'
        end as split
    from {{ ref('stg_transactions') }}
    where transaction_type = 'buy'
),
first_buy as (
    select customer_id, isin, min(transaction_date) as first_buy_date
    from buys
    group by all
)
select
    b.customer_id,
    b.isin,
    b.split,
    count(*)                       as n_buys,
    sum(b.total_value)             as total_value,
    min(b.transaction_date)        as first_date,
    max(b.transaction_date)        as last_date,
    min(b.transaction_date) = any_value(f.first_buy_date) as is_new
from buys b
join first_buy f using (customer_id, isin)
where b.split is not null
group by b.customer_id, b.isin, b.split
