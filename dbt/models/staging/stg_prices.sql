-- Daily close prices in euros, one row per asset and day.
select
    ISIN                                     as isin,
    cast("timestamp" as date)                as price_date,
    cast(closePrice as double)               as close_price
from {{ source('far_trans', 'close_prices') }}
where closePrice is not null and closePrice > 0
qualify row_number() over (partition by ISIN, "timestamp" order by closePrice) = 1
