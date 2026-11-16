-- First and last priced day per asset, used to decide which assets are evaluable at a cutoff.
select
    isin,
    min(price_date)  as first_price_date,
    max(price_date)  as last_price_date,
    count(*)         as n_prices
from {{ ref('stg_prices') }}
group by isin
