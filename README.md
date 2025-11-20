# Premier League 2026 Win Probability Model

A quantitative analysis of the 2026 Premier League season using historical team-level statistics (2016–2025) to estimate championship win probabilities.

## Overview

I built a predictive model trained on team performance metrics (goals, expected goals, defensive actions, etc.) across 10 seasons. Using rolling-window validation on 7 folds, I achieved **98.57% average test accuracy** with an XGBoost classifier. I then trained on the final three seasons (2023–2025) and generated win probabilities for all 20 Premier League teams.

## Key Results

| Team | My Probability (%) | Ask (decimal) | Bid (decimal) |
|---|---:|---:|---:|
| **Arsenal** | 52.59 | 1.90 | 1.86 |
| Manchester City | 14.45 | 6.92 | 6.78 |
| Chelsea | 5.71 | 17.52 | 17.17 |
| Fulham | 2.57 | 38.92 | 38.14 |
| Sunderland | 2.55 | 39.14 | 38.36 |
| Liverpool | 2.24 | 44.72 | 43.83 |
| *Others* | ~8.77 | — | — |

**Arsenal is my clear favorite** at 52.59% win probability. The top 3 teams (Arsenal, City, Chelsea) account for ~73% of the total implied probability mass.

## Market Quotes (Bid/Ask)

- **Ask** (what I'm selling): 1 / my probability. Anyone can back a team at the ask price against my book.
- **Bid** (what I'm buying): Ask × 0.98. Anyone can take my bid and bet with me.
- **Spread**: 2% (bid/ask margin).

## Model & Validation

### Features (20 total)
- **Team aggregates**: Goals (G), Expected Goals (xG), Defensive Clearances (DC), Passes per Defensive Action (OPDA), etc.
- **Engineered features**: Goal Difference (GD), Points Per Match (PPM), xG Differential (xGD).

### Rolling Window Validation
- **Strategy**: Train on 3 consecutive seasons → test on the next season.
- **Folds**: 7 windows (2016–2018→2019, 2017–2019→2020, …, 2023–2025→2026).
- **Performance**:
  - XGBoost: 98.57% avg test accuracy, 100% ROC AUC.
  - MLP (neural network): 95.71% avg test accuracy (XGBoost better for small datasets).
  - Train/test gap: −0.56% (no overfitting).

### Monte Carlo Simulation
- For each test season, I simulated 10,000 complete EPL seasons (380 matches each) using my predicted match probabilities.
- Empirical win frequencies converge to my model probabilities over the simulations.

## Files

- **`main.py`**: Full pipeline (data fetch, feature engineering, model training, validation, Monte Carlo, odds generation).
- **`2026_predictions.csv`**: Final win probabilities for all 20 teams.
- **`2026_pre_vs_inseason_comparison.csv`**: Pre-season vs. current in-season probability updates.
- **`RESULTS.md`**: Detailed breakdown of model performance, validation results, and 2026 predictions.

## How to Use

1. **View my quotes**: See the table above or open `2026_predictions.csv`.
2. **Accept the ask**: Back a team at my quoted odds against my book.
3. **Take my bid**: Bet with me at the bid price.
4. **Run the model**: `python main.py` to regenerate predictions and Monte Carlo simulations.

## Technical Notes

- Data source: Understat (team-level metrics, 2016–2025).
- Player-level data: Not available from Understat API; analysis uses team aggregates only.
- Model: XGBoost (gradient boosted trees). Compared against MLP for robustness.
- Validation: Time-series rolling windows to avoid look-ahead bias.
- No overfitting: Test accuracy matches training accuracy across all folds.

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
aiohttp
nest_asyncio
understat
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost aiohttp nest_asyncio understat
```

## Run

```bash
python main.py
```

---

**Status**: Production. Arsenal favored; liquidity available on all 20 teams.
