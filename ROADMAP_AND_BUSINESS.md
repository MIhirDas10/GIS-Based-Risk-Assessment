# Dengue Risk Prediction — Roadmap & Business Model

*A strategy document: what's built, what's missing, where it could go, and how to make it a sustainable product rather than a throwaway portfolio piece.*

*Last updated: 2026-05-26*

---

## 0. The one-line pitch

> **Climate-driven dengue outbreak early-warning for Bangladesh** — district-level weekly case forecasts that turn free satellite weather data into 2–4 weeks of lead time for public-health decisions.

The technical asset already exists. The question this doc answers: **who needs it, who pays, and how do we keep it alive?**

---

## 1. ✅ What has been DONE

### Data & pipeline (Days 1–7)
- [x] 7-container Docker stack (PostGIS, Redis, MLflow, Airflow ×3, API, dashboard) with `restart: always`
- [x] PostGIS schema with 5 schemas + 2 databases, auto-bootstrapped
- [x] 4 ingestion modules, all working: boundaries (64 districts), disease (856 weekly rows), population (WorldPop 2020), **real ERA5 weather (2019→2026, all 64 districts)**
- [x] 5 Airflow DAGs orchestrating ingestion + feature build
- [x] Feature mart: 7-step SQL CTE pipeline, 18 ML features (lags, rolling, interactions, spatial lag, cyclical, hotspot)
- [x] 4 models trained (Seasonal Naive, Ridge, XGBoost, LightGBM) with temporal split (no leakage)
- [x] MLflow registry with auto-promotion (XGBoost in **Production**, RMSE 1129, R² 0.286, Outbreak F1 0.634)
- [x] Prediction pipeline → GeoJSON maps + Redis cache + DB prediction log

### Serving layer (Days 8–9)
- [x] FastAPI REST service, 7 endpoints (`/health`, `/metrics/model`, `/districts`, `/district/{id}`, `/risk/current`, `/risk/forecast/{weeks}`, `/docs`)
- [x] Streamlit dashboard, 3 tabs (Risk Map, District Detail, Model), Folium choropleth, editorial UI
- [x] **`forecast_mart`** — serves predictions for the current real week (2026-W09) across all 64 districts, not just the 8 with historical case data

### Engineering quality (partial Days 11–12)
- [x] 118 automated tests (unit + integration), all passing
- [x] Ruff lint config + clean codebase
- [x] README, MODEL_CARD, DATA_DICTIONARY, STATUS_REPORT
- [x] GitHub Actions CI workflow (⚠️ needs a fix — see §2)
- [x] 16 real bugs found & fixed during verification

---

## 2. ❌ What is NOT done

### Day 10 — Monitoring & auto-retrain (not started)
- [ ] `src/monitoring/monitor.py` — Evidently data-drift + prediction-vs-actual tracking
- [ ] `airflow/dags/dag_monitor.py` — weekly drift check
- [ ] `airflow/dags/dag_retrain.py` — auto-retrain when RMSE degrades >15%
- [ ] `evidently` is in requirements but unused

### Day 11–12 — Final polish (remaining)
- [ ] `IMPACT.md`
- [ ] Fix CI: it installs only `ruff pytest` (tests import pandas/fastapi/streamlit → will fail); pins Python 3.10 vs the stack's 3.11
- [ ] Git commit + merge `develop` → `main` (everything is uncommitted)

### Orchestration gaps
- [ ] `forecast_mart` build not wired into a DAG (run manually via `--forecast`)
- [ ] No prediction DAG (predict.py runs manually; its docstring references an orchestration DAG that doesn't exist)

### The big honest gap: **model accuracy & recency**
- [ ] **R² 0.286 / Outbreak F1 0.63 is "interesting demo," not "decision-grade."** Nobody pays for a 63%-accurate outbreak flag yet.
- [ ] **Disease ground truth stops at 2023.** Predictions for 2026 use last-year spatial-lag stand-ins — fine for a demo, not validated against reality.
- [ ] **3-month weather lag** (ERA5-Land) — "current" is really ~12 weeks ago.

These three are the difference between a project and a product. Everything in §4 and §5 hinges on closing them.

---

## 3. What more we COULD do (technical roadmap)

### Tier 1 — Quick wins (days)
| Item | Why it matters |
|---|---|
| Finish Day 10 (drift + retrain DAGs) | Makes it self-maintaining → "set and forget" credibility |
| Fix CI + commit everything | Basic professionalism; protects the work |
| Email/SMS/Telegram alerts when a district hits Critical | Turns a dashboard into a *service* people actually use |
| Add a "model accuracy over time" page | Trust. Buyers need to see the track record. |

### Tier 2 — Makes it credible (weeks)
| Item | Why it matters |
|---|---|
| **Scrape DGHS weekly bulletins for 2024–2026 case data** | The single highest-leverage task. Real recent ground truth → retrain → validate → an accuracy story you can sell |
| Swap/augment ERA5-Land with **GFS or open-meteo forecast** (6-hr lag) | True real-time + genuine *future* weather instead of persistence assumptions |
| Add features: Google/Apple mobility, school calendar, Aedes-favorable-days index, prior-year outbreak phase | Push R² up — this is where accuracy gains live |
| Backtesting harness: "if you'd used this in 2023, how many outbreak-weeks would you have caught X weeks early?" | The killer sales slide |

### Tier 3 — Ambitious / differentiating (months)
| Item | Why it matters |
|---|---|
| **Multi-disease** (chikungunya, malaria, cholera) on the same engine | 4× the market for ~1.3× the work |
| **Multi-country** (start with dengue-endemic neighbors: India, Pakistan, Philippines, Brazil) | The architecture is country-agnostic; data sources are global (ERA5, WorldPop are worldwide) |
| Probabilistic forecasts (quantile regression / conformal prediction) | "70% chance of >500 cases" is far more actionable than a point estimate — and required for insurance |
| Hospital-level demand model (beds, IV fluids, blood platelets) | Converts an epidemiology tool into an operations tool people budget for |
| Spatial spread model (graph neural net on the district adjacency) | Genuinely novel; publishable; defensible |

---

## 4. Business model options (survey)

The same prediction engine can be sold to very different buyers. Scoring each on **willingness to pay**, **sales difficulty**, and **fit with what's built**.

| # | Model | Buyer | Pays for | $ | Ease | Notes |
|---|---|---|---|---|---|---|
| 1 | **B2G SaaS** | DGHS, city corporations | Outbreak early warning → vector-control & hospital planning | ●●○ | ●○○ | Highest mission fit; brutal sales cycle, tight budgets, procurement bureaucracy |
| 2 | **Grant / NGO-funded** | WHO, icddr,b, BRAC, Gates, Wellcome | Pilots, research, "impact" | ●●○ | ●●○ | Realistic *first* revenue; non-dilutive; but not recurring/scalable alone |
| 3 | **B2B hospital SaaS** | Private chains (Square, Evercare, United) | Surge staffing & supply planning | ●●● | ●●○ | They have money + clear ROI (a missed surge costs them); underrated |
| 4 | **Parametric insurance / reinsurance** | Insurers, Swiss Re, Munich Re, ADB | Outbreak risk pricing & payout triggers | ●●● | ●○○ | Biggest $; needs probabilistic + validated model; long lead but huge |
| 5 | **Pharma/FMCG demand intel** | Reckitt (Mortein), test-kit & IV-fluid makers | Demand forecasting, targeted campaigns | ●●● | ●●○ | They already spend on this; dengue season = their sales season |
| 6 | **Data/API subscription** | Researchers, agritech, other startups | Clean climate-health data feed | ●○○ | ●●● | Easy to ship, low $, good top-of-funnel |
| 7 | **B2C alerts app** | Citizens | Peace of mind, prevention nudges | ●○○ | ●●○ | Hard to monetize in BD directly; great for distribution/brand, ad/sponsor-supported |

(● = higher is better for $ and ease)

---

## 5. ⭐ Recommended model: **"Climate-Health Intelligence" — grant-seeded, then B2B2G subscription + insurance data licensing**

Don't pick one box. Sequence them so each stage funds the next and de-risks the one after.

### The wedge (Phase 1, 0–6 months): **Grant-funded public-good pilot**
- Partner with **icddr,b** (Bangladesh's premier health-research institute) or apply to **Wellcome Trust / Gates Foundation climate-health calls**.
- Deliverable: a validated dengue early-warning system DGHS can actually use, free during the pilot.
- **Why first:** non-dilutive money, gives you the one thing you can't buy — *real recent data + a credibility partner + a validation track record*. This directly closes the §2 accuracy gap.

### The engine (Phase 2, 6–18 months): **B2B subscription**
Two paying segments, same product, minimal extra build:
1. **Private hospital chains** — "Dengue Surge Planner": 3-week-ahead admissions forecast per facility catchment. Clear ROI: one prevented stockout or one right-sized staffing week pays the subscription. Sell as seats/dashboard + alerts. ~$500–2,000/mo per chain.
2. **FMCG/pharma** — "Dengue Season Intelligence": district-level demand signals for repellents, test kits, IV fluids, paracetamol. Annual data-licensing deal. This is the cash cow — Reckitt-type companies already budget for market intelligence.

### The moat (Phase 3, 18+ months): **Parametric-insurance data licensing**
- License validated probabilistic outbreak forecasts to insurers/reinsurers/ADB for parametric dengue insurance (auto-payout when forecast/actuals cross a threshold).
- Highest margin, most defensible — but only credible *after* you have a multi-year validated track record from Phases 1–2.

### Why this sequence is sustainable (not a flash-in-the-pan)
- **Data network effect:** every week of operation adds ground truth → better model → more valuable to every segment. Competitors starting later can't buy your history.
- **Same engine, many buyers:** the marginal cost of adding a hospital or an FMCG client is near zero.
- **Mission + money aligned:** the public-good version (saving lives) *generates the data* that powers the commercial versions. The grant work isn't charity — it's R&D + distribution.
- **Geographic & disease expansion is a copy-paste:** ERA5 and WorldPop are global; the pipeline is country-agnostic. Bangladesh dengue → SE-Asia dengue → malaria/chikungunya/cholera is mostly config, not rewrite.

---

## 6. How to implement the business model (concrete)

### Step 0 — Make it undeniable (now)
1. Close the accuracy gap: scrape 2024–26 DGHS data, retrain, and **produce the backtest slide**: *"X weeks of lead time, Y% of outbreak-weeks caught."* No backtest = no sale.
2. Ship alerting (email/Telegram) — a forecast nobody is notified about has no value.
3. Add the "accuracy over time" page so the track record is visible and self-documenting.

### Step 1 — Get the credibility partner (months 1–3)
- Cold-email icddr,b's dengue/epidemiology group and 1–2 academic epidemiologists. Offer the tool free for a season in exchange for data access + a validation co-authorship.
- Apply to 2–3 climate-health grants (Wellcome, Gates, ADB, Grand Challenges).

### Step 2 — Land one paying logo per segment (months 3–9)
- One private hospital chain pilot (free → paid after one season proves ROI).
- One FMCG/pharma data-licensing conversation (they move fast when it's tied to a sales season).

### Step 3 — Productize & price (months 6–12)
- Tiers: **Free** (public risk map, builds trust + distribution) → **Pro** (alerts, API, per-facility forecasts) → **Enterprise** (custom catchments, SLA, data license).
- Stand up billing (Stripe), usage metering on the API (you already have the API + Redis to meter).

### Step 4 — Expand (12+ months)
- Second disease (chikungunya — same Aedes vector, almost free to add).
- Second country (pick one with open case data + a partner).

### What to build vs. what to sell
- **Build:** alerting, backtest harness, accuracy page, probabilistic forecasts, per-facility catchment aggregation.
- **Don't build yet:** mobile app, multi-tenant billing portal, ML for diseases you have no data for. Sell the dashboard + API + a PDF report first; automate after someone pays.

---

## 7. Honest risks & how to counter them

| Risk | Counter |
|---|---|
| Model isn't accurate enough to trust | Phase 1 is *about* fixing this with real data + validation before charging anyone |
| B2G sales are slow and underfunded | That's why government is the *mission/data* play, not the *revenue* play — revenue comes from hospitals/FMCG/insurance |
| Data dependency (DGHS could close the tap) | Diversify sources early (icddr,b partnership, hospital admissions data, news-based nowcasting); make the relationship mutual |
| "Free government tool will undercut us" | Position as the *operations layer* (alerts, per-facility, SLA, integration) that a government dashboard never provides |
| Solo-founder bandwidth | Grant money buys time; pick ONE commercial segment to chase first (recommend hospitals — fastest ROI story) |
| Climate/weather lag kills "real-time" claim | Add forecast weather (GFS/open-meteo) so you predict *forward* from real future weather, not persistence |

---

## 8. Why this is interesting (the part that keeps you motivated)

- It's **real**: live data, auto-updating, a genuine public-health problem that kills people every monsoon.
- It's **defensible**: the moat is data + validation track record, which compounds over time.
- It's **expandable**: one engine → many diseases → many countries → many buyers.
- It's **dual-purpose**: the same code saves lives *and* can fund itself — you're not choosing between impact and sustainability.
- It's **publishable**: the spatial-spread + early-warning validation is genuine research; a paper is both credibility and marketing.

The thing that makes projects "flush away" is that they're static and serve nobody. This one updates itself weekly, notifies a real decision-maker, and gets smarter with every season. That's the opposite of throwaway — **if** you close the accuracy gap and put it in front of one real user.

---

## 9. Recommended immediate next 3 actions

1. **Scrape 2024–2026 DGHS case data → retrain → produce the backtest.** (Turns the demo into evidence.)
2. **Finish Day 10 (drift + retrain) and ship Critical-tier alerts.** (Turns the dashboard into a service.)
3. **Email icddr,b / one private hospital with the backtest slide.** (Turns the service into a conversation.)

Everything else follows from having (a) proof it works and (b) one real user who cares.
