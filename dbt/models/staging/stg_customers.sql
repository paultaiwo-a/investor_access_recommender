-- Customer profile history as a slowly changing dimension (valid_from, valid_to).
-- risk_level drops the "Predicted_" prefix and keeps a flag for it.
with base as (
    select distinct
        customerID                           as customer_id,
        customerType                         as customer_type,
        riskLevel                            as risk_level_raw,
        investmentCapacity                   as investment_capacity_raw,
        cast(lastQuestionnaireDate as date)  as last_questionnaire_date,
        cast("timestamp" as date)            as valid_from
    from {{ source('far_trans', 'customer_information') }}
)
select
    customer_id,
    customer_type,
    replace(risk_level_raw, 'Predicted_', '')              as risk_level,
    starts_with(risk_level_raw, 'Predicted_')              as risk_level_is_predicted,
    replace(investment_capacity_raw, 'Predicted_', '')     as investment_capacity,
    last_questionnaire_date,
    valid_from,
    lead(valid_from) over (partition by customer_id order by valid_from) as valid_to
from base
qualify row_number() over (partition by customer_id, valid_from order by risk_level_raw) = 1
