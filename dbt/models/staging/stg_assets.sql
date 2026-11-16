-- One row per asset (latest description). asset_group is the bucket the sector cap uses:
-- the economic sector when known, otherwise category and sub-category.
with ranked as (
    select
        *,
        row_number() over (partition by ISIN order by "timestamp" desc) as rn
    from {{ source('far_trans', 'asset_information') }}
)
select
    ISIN                                     as isin,
    assetName                                as asset_name,
    assetShortName                           as asset_short_name,
    assetCategory                            as asset_category,
    coalesce(assetSubCategory, 'Unspecified') as asset_subcategory,
    marketID                                 as market_id,
    sector,
    industry,
    coalesce(
        sector,
        assetCategory || ': ' || coalesce(assetSubCategory, 'Unspecified')
    )                                        as asset_group,
    cast("timestamp" as date)                as updated_at
from ranked
where rn = 1
