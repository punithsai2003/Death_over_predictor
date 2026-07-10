"""
Death-Over Matchup Advisor - Streamlit app.

Give it the batter, your available bowlers, and the match situation
(e.g. 30 needed off 12) - it tells you who should bowl which over.

Needs: models/matchup_app_bundle.joblib   (from Colab Cell 4)
Run:   streamlit run app_matchup.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path("models/matchup_app_bundle.joblib")

st.set_page_config(page_title="Death-Over Matchup Advisor", page_icon="🎯", layout="wide")
st.title("🎯 Death-Over Matchup Advisor")
st.caption("Who should bowl the 19th and the 20th? Probabilities from 20,000+ "
           "historical death overs (IPL + T20I, Cricsheet).")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


bundle = load_bundle()
models, FEATURES = bundle["models"], bundle["features"]
bat_stats, bowl_stats = bundle["bat_stats"], bundle["bowl_stats"]

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Situation")
    batter = st.selectbox("Batter on strike", sorted(bat_stats.index))
    bowler_a = st.selectbox("Bowler option A", sorted(bowl_stats.index))
    bowler_b = st.selectbox("Bowler option B", sorted(bowl_stats.index),
                            index=min(1, len(bowl_stats) - 1))
    runs_needed = st.number_input("Runs needed", 1, 60, 30)
    balls_remaining = st.selectbox("Balls remaining", [12, 6], index=0)
    st.caption("12 balls = deciding overs 19 & 20 · 6 balls = final over only")


def predict(batter, bowler, over, need, balls):
    X = pd.DataFrame([{
        "over": over, "innings": 2, "is_chase": 1,
        "runs_needed": need, "balls_remaining": balls,
        **bat_stats.loc[batter].to_dict(),
        **bowl_stats.loc[bowler].to_dict(),
    }])[FEATURES]
    return (models["batter_12plus"].predict_proba(X)[0, 1],
            models["bowler_max8"].predict_proba(X)[0, 1])


if st.button("Advise", type="primary", use_container_width=True):
    b = bat_stats.loc[batter]
    st.write(f"**{batter}** at the death: SR {b.bat_death_sr}, "
             f"boundary {b.bat_boundary_pct}%, dot {b.bat_dot_pct}%")
    st.divider()

    if balls_remaining == 6:
        # ---- final over: simple head-to-head ----
        st.subheader(f"Final over — defending {runs_needed}")
        rows = []
        for bw in (bowler_a, bowler_b):
            p12, p8 = predict(batter, bw, 20, runs_needed, 6)
            rows.append({"Bowler": bw, "P(batter 12+)": f"{p12:.0%}",
                         "P(concedes ≤8)": f"{p8:.0%}", "_p12": p12})
        table = pd.DataFrame(rows)
        st.dataframe(table.drop(columns="_p12"), use_container_width=True, hide_index=True)
        best = table.sort_values("_p12").iloc[0]["Bowler"]
        st.success(f"**Recommendation: {best} bowls the final over** "
                   f"(lower probability of a big over against {batter}).")
    else:
        # ---- overs 19 & 20: compare both assignments ----
        st.subheader(f"Overs 19 & 20 — defending {runs_needed} off 12")
        need_20 = max(runs_needed - 12, 1)   # assumed state entering over 20

        rows, risk = [], {}
        for bw in (bowler_a, bowler_b):
            p12_19, p8_19 = predict(batter, bw, 19, runs_needed, 12)
            p12_20, p8_20 = predict(batter, bw, 20, need_20, 6)
            risk[bw] = {"19": p12_19, "20": p12_20}
            rows.append({"Bowler": bw,
                         "Over 19 P(12+)": f"{p12_19:.0%}",
                         "Over 19 P(≤8)": f"{p8_19:.0%}",
                         "Over 20 P(12+)": f"{p12_20:.0%}",
                         "Over 20 P(≤8)": f"{p8_20:.0%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # total risk of each assignment = sum of P(12+) across both overs
        plan1 = risk[bowler_a]["19"] + risk[bowler_b]["20"]   # A->19, B->20
        plan2 = risk[bowler_b]["19"] + risk[bowler_a]["20"]   # B->19, A->20
        if plan1 <= plan2:
            first, last, total = bowler_a, bowler_b, plan1
        else:
            first, last, total = bowler_b, bowler_a, plan2
        st.success(f"**Recommendation: {first} bowls over 19, {last} bowls over 20** "
                   f"(combined big-over risk {total:.0%} vs {max(plan1, plan2):.0%} "
                   f"for the reverse).")
        st.caption("Over-20 state assumes ~12 conceded in over 19; the model uses the "
                   "striker at the start of each over (strike rotation not simulated).")

st.markdown("---")
st.caption("Model: gradient boosting on chronologically-built player histories "
           "(no future leakage) | AUC ~0.64 — this shifts odds, it doesn't see the future. "
           "Built by Punith Sai P · Data © Cricsheet")