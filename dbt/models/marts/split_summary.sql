-- Sanity table: size of each split, used in the README.
select
    split,
    count(*)                                   as n_pairs,
    count(*) filter (where is_new)             as n_new_pairs,
    count(distinct customer_id)                as n_customers,
    count(distinct isin)                       as n_assets,
    min(first_date)                            as min_date,
    max(last_date)                             as max_date
from {{ ref('interactions') }}
group by split
