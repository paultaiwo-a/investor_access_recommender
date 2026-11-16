"""Suitability layer applied after ranking.

Two rules, in the spirit of MiFID II suitability and concentration guidance:

1. **Risk fit.** Drop any asset whose SRRI class exceeds what the customer's risk profile
   allows (see :mod:`recs.rules.risk`).
2. **Concentration cap.** No single sector (``asset_group``) may take more than
   ``sector_cap`` of the final top-k (40% of 10 = at most 4 assets).

The layer walks the ranked candidate list in order, skipping assets that break a rule, and
backfills from deeper candidates so the customer still gets k suggestions when possible.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from recs.rules.risk import MAX_SRRI, max_srri


@dataclass
class RuleOutcome:
    items: list[str]
    removed: list[tuple[str, str]] = field(default_factory=list)  # (isin, reason)


@dataclass
class SuitabilityRules:
    k: int = 10
    sector_cap: float = 0.4
    enforce_risk: bool = True
    enforce_sector_cap: bool = True
    max_srri_by_profile: Mapping[str, int] = field(default_factory=lambda: dict(MAX_SRRI))

    @property
    def max_per_group(self) -> int:
        return max(1, math.floor(self.sector_cap * self.k))

    def allowed_srri(self, risk_level: str | None) -> int:
        if risk_level in self.max_srri_by_profile:
            return self.max_srri_by_profile[risk_level]
        return max_srri(risk_level)

    def apply(
        self,
        ranked: Sequence[str],
        risk_level: str | None,
        srri: Mapping[str, float],
        group: Mapping[str, str],
    ) -> RuleOutcome:
        limit = self.allowed_srri(risk_level)
        out: list[str] = []
        removed: list[tuple[str, str]] = []
        per_group: Counter[str] = Counter()
        for isin in ranked:
            if len(out) == self.k:
                break
            if self.enforce_risk and srri.get(isin, 7) > limit:
                removed.append((isin, "risk"))
                continue
            g = group.get(isin, "Unknown")
            if self.enforce_sector_cap and per_group[g] >= self.max_per_group:
                removed.append((isin, "sector_cap"))
                continue
            per_group[g] += 1
            out.append(isin)
        return RuleOutcome(out, removed)

    def apply_all(
        self,
        ranked: Mapping[str, Sequence[str]],
        risk_levels: Mapping[str, str],
        item_meta: pd.DataFrame,
    ) -> dict[str, RuleOutcome]:
        """``item_meta`` is indexed by ISIN with ``srri`` and ``asset_group`` columns."""
        srri = item_meta["srri"].to_dict()
        group = item_meta["asset_group"].to_dict()
        return {
            u: self.apply(items, risk_levels.get(u), srri, group) for u, items in ranked.items()
        }


def violations(
    recs: Mapping[str, Sequence[str]],
    risk_levels: Mapping[str, str],
    item_meta: pd.DataFrame,
    rules: SuitabilityRules | None = None,
) -> dict[str, float]:
    """Share of top-k lists that break each rule (before the layer is applied)."""
    rules = rules or SuitabilityRules()
    srri = item_meta["srri"].to_dict()
    group = item_meta["asset_group"].to_dict()
    risk_bad = cap_bad = items_bad = n_items = 0
    for u, items in recs.items():
        top = list(items)[: rules.k]
        limit = rules.allowed_srri(risk_levels.get(u))
        bad = [i for i in top if srri.get(i, 7) > limit]
        risk_bad += bool(bad)
        items_bad += len(bad)
        n_items += len(top)
        counts = Counter(group.get(i, "Unknown") for i in top)
        cap_bad += any(c > rules.max_per_group for c in counts.values())
    n = max(len(recs), 1)
    return {
        "lists_with_unsuitable_risk": risk_bad / n,
        "unsuitable_item_share": items_bad / max(n_items, 1),
        "lists_breaking_sector_cap": cap_bad / n,
    }
