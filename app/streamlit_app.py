"""Streamlit demo: a "Discover" tab for one customer.

Reads the same serving artifacts as the API. Run with ``make app``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import DEFAULT_DIR, DISCLAIMER, Store  # noqa: E402

RISK_COLORS = {
    1: "#2e7d32",
    2: "#558b2f",
    3: "#9e9d24",
    4: "#f9a825",
    5: "#ef6c00",
    6: "#d84315",
    7: "#b71c1c",
}


@st.cache_resource
def load_store() -> Store:
    return Store(Path(os.environ.get("RECS_SERVING_DIR", DEFAULT_DIR)))


def risk_badge(cls: int) -> str:
    return (
        f"<span style='background:{RISK_COLORS.get(cls, '#555')};color:white;"
        f"padding:2px 8px;border-radius:10px;font-size:0.8em'>Risk {cls}/7</span>"
    )


def main() -> None:
    st.set_page_config(page_title="Discover: assets you might like", layout="wide")
    try:
        store = load_store()
    except FileNotFoundError:
        st.error("No serving artifacts found. Run `make serving` first.")
        return

    st.title("Discover")
    st.caption(
        f"Stocks and funds you might like, as of {store.meta['as_of']}. "
        "Built on FAR-Trans (National Bank of Greece retail transactions)."
    )

    with st.sidebar:
        st.header("Customer")
        levels = ["Any", *sorted(store.customers["risk_level"].unique())]
        lvl = st.selectbox("Risk profile", levels)
        pool = (
            store.customers
            if lvl == "Any"
            else store.customers[store.customers["risk_level"] == lvl]
        )
        pool = pool[pool.index.isin(store.recs.keys())]
        sample = pool.sort_values("n_past_assets", ascending=False).head(200).index.tolist()
        cid = st.selectbox("Customer ID", sample) if sample else None
        typed = st.text_input("...or paste a customer ID")
        cid = typed.strip() or cid
        k = st.slider("How many suggestions", 1, 10, 10)

    if not cid:
        st.info("Pick a customer in the sidebar.")
        return
    try:
        resp = store.recommend(cid, k)
    except KeyError:
        st.warning("Unknown customer ID.")
        return

    c = store.customers.loc[cid]
    holdings = json.loads(c["holdings"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk profile", resp.risk_level)
    m2.metric("Highest risk class allowed", f"{resp.max_risk_class} / 7")
    m3.metric("Current holdings", len(holdings))
    m4.metric("Strategy", resp.strategy.replace("_", " "))

    if holdings:
        with st.expander("Current holdings"):
            h = store.assets.reindex(holdings)[["asset_name", "asset_category", "sector", "srri"]]
            st.dataframe(h.rename(columns={"srri": "risk class"}), use_container_width=True)

    st.subheader("Suggested for you")
    for rec in resp.recommendations:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(
                f"**{rec.rank}. {rec.name}**  \n`{rec.isin}` · {rec.category}"
                + (f" · {rec.subcategory}" if rec.subcategory else "")
                + (f" · {rec.sector}" if rec.sector else "")
            )
            for r in rec.reasons:
                left.markdown(f"- {r}")
            right.markdown(risk_badge(rec.risk_class), unsafe_allow_html=True)
            if rec.return_6m is not None:
                right.metric("6-month return", f"{rec.return_6m * 100:+.1f}%")

    sectors = pd.Series([r.sector or r.category for r in resp.recommendations]).value_counts()
    st.caption("Sector mix of the list (no sector may exceed 40% of the top 10):")
    st.bar_chart(sectors)
    st.caption(DISCLAIMER)


main()
