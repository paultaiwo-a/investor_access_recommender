select
    exchangeID   as exchange_id,
    marketID     as market_id,
    name         as market_name,
    country,
    marketClass  as market_class
from {{ source('far_trans', 'markets') }}
