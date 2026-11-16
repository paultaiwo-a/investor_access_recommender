# Online test plan: two-stage ranker vs popularity in the Discover tab

This plan takes the offline winner (the two-stage LambdaRank ranker) to an online A/B test
against the popularity baseline. The readout reuses `abkit` from the
[deposit-nudge-experiment](https://github.com/paultaiwo-a/deposit_nudge_experiment) project
(installed here as the optional `experiment` extra), and the power calculation follows that
project's `scripts/power_analysis.py` (statsmodels `NormalIndPower` on an arcsine effect size).
Inputs come from `python -m recs.pipelines.ab_power`, saved in `results/ab_power.json`.

## 1. Hypothesis

Customers shown ranker suggestions in Discover buy a suggested asset more often than customers
shown the most popular assets, with no increase in unsuitable exposure, regret selling or
portfolio concentration.

Offline evidence (test window, both arms under the same suitability rules):

| | Popularity + rules | Ranker + rules |
|---|---:|---:|
| 30-day first-time buy of a suggested asset, active customers (n = 4,732) | 0.74% | 0.95% |
| Relative lift | | +29% (abkit z-test p = 0.26) |
| nDCG@10 against suitable purchases (6-month window) | 0.180 | 0.359 |

The offline rate is organic (nobody saw the list), so the online rate should be higher in both
arms. The +29% gap is a prior, not a forecast. The plan is sized for a smaller effect.

## 2. Design

| Item | Choice |
|---|---|
| Unit | Customer (randomise once, sticky across sessions and devices) |
| Population | Customers with at least one past buy who open Discover during the test. Cold-start customers get the profile fallback in both arms and are excluded from the primary analysis |
| Arms | **Control:** popularity, then the suitability layer. **Treatment:** two-stage ranker, then the same suitability layer. Same UI, same 10 slots, same reasons format (control reasons are "Popular with customers") |
| Split | 50/50, hash of customer ID with an experiment salt |
| Trigger | Exposure is logged when the Discover list renders. Analysis is on exposed customers only (triggered analysis), assignment is still at customer level |
| Duration | Whole weeks, minimum 4, sized by the power table below. Pre-registered end date, no early stop on the primary metric except via the sequential test in section 6 |
| Batch | Both arms are scored by the same daily batch (`python -m recs.serving`), so freshness is equal |

Both arms run the suitability layer. The test compares rankers, not rankers with and without
compliance. Shipping without the layer is not an option for an advised list.

## 3. Metrics

**Primary (decision metric)**

* **Recommendation buy rate:** share of exposed customers who make a first-time buy of an asset
  that was in their shown list, within 30 days of first exposure. Binary per customer, read
  with `abkit.two_prop_ztest`.

**Secondary**

* Suggested-asset buys per exposed customer (ratio metric across sessions, `abkit.delta_method_ratio`).
* Euro invested in suggested assets per exposed customer (`abkit.welch_ttest`, variance reduced with
  `abkit.cuped_adjust` on the customer's pre-period invested amount).
* Click-through from Discover to an asset page (leading indicator, also a fallback primary if
  the buy rate is underpowered, see section 5).
* Catalog coverage and the share of buys that go to assets outside the 20 most popular.

Secondary metrics are corrected with `abkit.holm`.

**Guardrails (any breach blocks the launch)**

| Guardrail | Threshold |
|---|---|
| Unsuitable suggestions (risk class above profile) | Exactly 0 in both arms. Checked from the serving logs every day; any hit stops the test |
| Sector concentration | No list with more than 4 of 10 in one sector (logged) |
| Overall buy rate (all assets, not only suggested) | Not significantly lower in treatment (non-inferiority margin -5% relative). Catches cannibalisation |
| Regret proxy: sells of a suggested asset within 30 days of buying it | Not significantly higher in treatment |
| Portfolio concentration (HHI of holdings by sector) after 30 days | Not significantly higher in treatment |
| Complaints and Discover hide/opt-out rate | Not significantly higher in treatment |
| Sample ratio mismatch | `abkit.srm_check` p > 0.001 on exposed counts, daily. A failure invalidates the readout |

**Monitored, not decision metrics**

* 6-month return of suggested assets (the FAR-Trans profitability@k). Too slow and too driven by
  the market to decide on in a 4 to 8 week test, and the product should not be tuned to
  short-term returns. Tracked on a long-term holdout (section 7).

## 4. Power

Inputs from `results/ab_power.json`: control rate p0 = 0.74% (30-day organic rate of the
popularity arm with rules), alpha = 0.05 two-sided, power = 0.8, 50/50 split.

Minimum detectable effect by sample size per arm:

| Customers per arm | MDE (absolute) | MDE (relative) |
|---:|---:|---:|
| 2,366 (FAR-Trans active base, halved) | 0.86 pp | +116% |
| 5,000 | 0.56 pp | +75% |
| 10,000 | 0.38 pp | +51% |
| 25,000 | 0.23 pp | +31% |
| 50,000 | 0.16 pp | +22% |

Customers per arm needed for a given relative lift:

| Relative lift | Per arm | Total |
|---:|---:|---:|
| +10% | 220,990 | 441,980 |
| +15% | 100,473 | 200,946 |
| +20% | 57,771 | 115,542 |
| +30% | 26,774 | 53,548 |

**What this means.** FAR-Trans is a sample of 29,090 customers with about 1,600 buying in a given
month. At that size the test could only detect an effect larger than doubling, so it is not
worth running on that population. At bank scale the picture changes: with around 115,000
exposed customers the test detects a +20% lift, which sits below the offline +29% gap.

**If traffic is short**, in order of preference:

1. **CUPED on the primary metric**, using each customer's buys in the 90 days before exposure as
   the covariate (`abkit.cuped_adjust`). Past buying strongly predicts future buying in this
   data, so expect a meaningful variance cut. Measure the correlation in an A/A run first.
2. **Lengthen the attribution window** from 30 to 60 days, which raises p0 and lowers the
   required n roughly in proportion.
3. **Switch the primary to click-through** to the asset page, which has a far higher base rate.
   Keep the buy rate as a must-not-lose secondary.

## 5. Pre-launch checks

* A/A test for one week on the same pipeline: SRM, metric variance, CUPED correlation, logging.
* Offline replay of the treatment batch for the A/A week: 0 unsuitable suggestions, every list
  has 10 items, sector cap respected (`recs.rules.suitability.violations`).
* Exposure and purchase events joined by customer ID and ISIN in the warehouse (a dbt model
  `fct_rec_exposures` that mirrors `fct_assignments` in the deposit-nudge project).

## 6. Analysis and decision rule

* Primary: `abkit.two_prop_ztest` on exposed customers at the pre-registered end date.
* Daily monitoring uses `abkit.always_valid_pvalues` (mSPRT), so peeking does not inflate false
  positives. The test can stop early for harm on any guardrail, not for a win.
* **Ship** if the primary lift is significant and positive, and no guardrail is breached.
* **Iterate** if the primary is flat but clicks or euro invested improve: keep collecting
  exposure data for the next ranker version.
* **Do not ship** if any guardrail is breached, whatever the primary says.

Segments read after the fact (descriptive, Holm corrected): risk profile, customer type
(Mass vs Premium), investment capacity, and customers with fewer than 3 past assets. The
offline suitability report shows Aggressive investors lose almost nothing to the rules while
Income and Conservative investors see a very different, mostly fund and bond, list, so effects
may differ by profile.

## 7. After launch

* Keep a 5% long-term holdout on popularity for 6 months to read the profitability metric
  (6-month return of suggested assets bought) and the slower guardrails (regret selling,
  concentration).
* Feed exposure logs back into training. The current ranker learns only from organic buys,
  which is the main reason the offline lift is a weak forecast of the online one.

## 8. Risks and caveats

* **Offline-online gap.** Offline labels are organic purchases; the online metric is purchases
  after exposure. Popularity may benefit more from exposure than the offline numbers suggest
  because its suggestions are already familiar.
* **Market regime.** A 4 to 8 week window covers one market regime. Rolling offline results
  (README) show the ranker's lead holds across 2019 to 2022, which lowers but does not remove
  this risk.
* **Interference.** Customers in the two arms trade in the same markets, but at retail volumes
  the effect on prices is negligible.
* **Regulation.** Both arms must carry the "information only, not personal advice" disclaimer,
  and the test needs sign-off from compliance before launch.
