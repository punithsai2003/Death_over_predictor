# Death-Over Matchup Advisor

A player-level machine learning tool that answers the single hardest question a T20 captain faces in the final overs:

> **Who bowls the 19th over — and to which batter?**

Give it a batter, two bowling options and the match state, and it returns a **risk matrix** and a **recommendation**, built from real IPL + T20 international history.

> **Live app:** "https://deathoverpredictor-6nuegnyzuqw7bzfrsnfu9t.streamlit.app/" · Built with Python · scikit-learn · Streamlit

---

## The question

Death overs decide T20 matches. A captain choosing between two bowlers for the 19th over is really asking: *given this batter, this match state and each bowler's history, which choice minimises the damage?* This tool turns that instinct into an evidence-based read.

## How it works

- **Two models, two targets** — one models the **batter's** likely output, one models the **bowler's** likely output, so the matchup is seen from both sides.
- **Chronological, leakage-free features** — player histories are built strictly **in time order**. Every feature for a given delivery is computed only from that player's *past*, never the future. No look-ahead, no leakage.
- **Inputs → output** — enter batter, two bowler options and the match state; the app returns a comparative **risk matrix** and a recommended matchup.

## Honest engineering

Single-over outcomes in cricket are genuinely **noisy** — a good ball can still go for six. This tool reports **AUC ≈ 0.64** and presents it for exactly what it is: a **real, odds-shifting edge**, not a crystal ball. Over a season of decisions, shifting the odds in your favour compounds. Reporting it honestly matters more than inflating it.

## At a glance

| Aspect | Detail |
|---|---|
| **Death overs modelled** | 20,000+ |
| **Data leakage** | 0 — features are strictly chronological |
| **Models** | 2 (batter target + bowler target) |
| **Competitions** | IPL + T20 Internationals |
| **Honest AUC** | ~0.64 (odds-shifting edge, reported as such) |

## Tech stack

- Python  - pandas  - scikit-learn - Gradient Boosting - time-aware feature engineering - Streamlit



## What I learned

- Building **time-aware features** so a model never accidentally sees the future — the core discipline behind trustworthy sports ML.
- Modelling a matchup from **both the batter's and bowler's** perspective.
- **Calibrating expectations**: presenting a modest-but-real edge honestly instead of chasing a misleading accuracy figure.


📧 punithsaipalakurthi@gmail.com · 💼 [LinkedIn](https://linkedin.com/in/punith-sai-p-8126b0215) · 🧑‍💻 [GitHub](https://github.com/punithsai2003)

*Ball-by-ball data © [Cricsheet](https://cricsheet.org/). This project is for analysis and educational purposes.*
