-- Load the FAR-Trans csvs and clean them up.
-- {{raw}} is the data/raw folder. The python script fills it in.

create or replace table transactions as
select
    cast(transactionID as bigint) as transaction_id,
    customerID as customer_id,
    ISIN as isin,
    lower(transactionType) as transaction_type,
    cast("timestamp" as date) as transaction_date,
    cast(totalValue as double) as total_value,
    cast(units as double) as units
from read_csv('{{raw}}/transactions.csv', header = true, auto_detect = true)
where customerID is not null
  and ISIN is not null
qualify row_number() over (partition by transactionID order by "timestamp") = 1;

create or replace table assets as
with ranked as (
    select
        *,
        row_number() over (partition by ISIN order by "timestamp" desc) as rn
    from read_csv('{{raw}}/asset_information.csv', header = true, auto_detect = true)
)
select
    ISIN as isin,
    assetName as asset_name,
    assetCategory as asset_category,
    sector
from ranked
where rn = 1;

create or replace table prices as
select
    ISIN as isin,
    cast("timestamp" as date) as price_date,
    cast(closePrice as double) as close_price
from read_csv('{{raw}}/close_prices.csv', header = true, auto_detect = true)
where closePrice is not null
  and closePrice > 0
qualify row_number() over (partition by ISIN, "timestamp" order by closePrice) = 1;

-- train: before 30 Nov 2021
-- val:   30 Nov 2021 up to 30 May 2022
-- test:  30 May 2022 up to 30 Nov 2022
create or replace table buys as
select
    customer_id,
    isin,
    transaction_date,
    total_value,
    case
        when transaction_date < date '2021-11-30' then 'train'
        when transaction_date < date '2022-05-30' then 'val'
        when transaction_date < date '2022-11-30' then 'test'
    end as split
from transactions
where transaction_type = 'buy';
