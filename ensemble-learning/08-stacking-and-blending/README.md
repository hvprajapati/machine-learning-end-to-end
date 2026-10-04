# 08 — Stacking and Blending

Voting gives every model an equal vote. **Stacking** trains a *meta-model* that learns **which base model to trust,
and how much**. **Blending** is the simpler hold-out version of the same idea. This folder teaches both, shows the
data-leakage trap of "naive" stacking, and measures honestly whether stacking helps on a small dataset.

## Files (read in this order)

| # | File | What it covers |
|---|---|---|
| 1 | [`theory.md`](theory.md) | Level-0 / level-1 models, democracy vs weighted analogy, why naive stacking overfits, blending and K-fold stacking algorithms with diagrams, numeric illustration, multi-layer stacking, choosing models, `passthrough`, `stack_method`, `StackingRegressor`, voting vs stacking vs blending table, common mistakes, revision questions |
| 2 | [`01-stacking.ipynb`](01-stacking.ipynb) | sklearn `StackingClassifier` on heart.csv: single models vs voting vs stacking with repeated cross-validation, meta-model coefficients, `passthrough` on/off, `StackingRegressor` on the diabetes dataset |
| 3 | [`02-blending-and-stacking-by-hand.ipynb`](02-blending-and-stacking-by-hand.ipynb) | Build naive stacking (leaky), blending and K-fold stacking yourself; check that hand-made K-fold stacking = sklearn's `StackingClassifier`; compare over 20 random splits |
| — | `heart.csv` | UCI heart-disease data: 303 patients, 13 features, `target` = 1 if heart disease |

## How to read the notebooks

- ✍️ **Core code: learn this** — the code you should understand and be able to write.
- 🎨 **Visualization only** — plotting code, collapsed. Run it and look at the picture.
- Each notebook ends with **Key takeaways** and a **🧠 Quick check** (click a question to see the answer).
- All results use fixed `random_state`s, so re-running gives the same numbers.

## Key results

**Heart disease, accuracy, 5-fold × 5 repeats cross-validation (`01-stacking.ipynb`)**

| model | mean acc | std |
|---|---|---|
| Logistic regression (scaled) — best single model | **0.829** | 0.043 |
| Stacking (RF + KNN + GBDT + LR → LR) | 0.824 | 0.047 |
| Random Forest | 0.822 | 0.044 |
| KNN (scaled) | 0.820 | 0.052 |
| Hard voting (RF + KNN + GBDT) | 0.820 | 0.041 |
| Soft voting (RF + KNN + GBDT) | 0.816 | 0.043 |
| Stacking (RF + KNN + GBDT → LR) | 0.815 | 0.043 |
| Gradient Boosting | 0.796 | 0.045 |

- Stacking did **not** beat the best single model; all differences are smaller than one standard deviation.
- Meta-model weights: KNN 2.50, RF 2.17, GBDT 0.47 → it trusts the weakest model (GBDT) least.
- `passthrough=True` 0.820 vs `False` 0.814 (within noise).
- Diabetes regression (R²): Ridge 0.479, StackingRegressor 0.473, GBR 0.459, RF 0.442, KNN 0.440.

**Naive vs blending vs K-fold stacking, mean over 20 random splits (`02-blending-and-stacking-by-hand.ipynb`)**

| recipe | train acc | test acc | test log-loss |
|---|---|---|---|
| Naive stacking (leaky) | 0.999 | 0.820 | 0.422 |
| Blending (hold-out) | — | 0.828 | 0.412 |
| K-fold stacking (= `StackingClassifier`) | 0.822 | **0.841** | **0.379** |
| KNN alone / RF alone | — | 0.841 / 0.839 | — |

Hand-made K-fold stacking gives exactly the same probabilities as sklearn's `StackingClassifier` with the same CV splitter.

## Learning objectives

After this folder you can:
- explain stacking (base models + meta-model) and how it differs from voting;
- explain why training the meta-model on in-sample predictions leaks, and spot it in code;
- implement blending and K-fold (out-of-fold) stacking by hand with `cross_val_predict`;
- use `StackingClassifier` / `StackingRegressor` with `cv`, `final_estimator`, `passthrough`, `stack_method`;
- read the meta-model's coefficients to see which base model it trusts;
- compare ensembles honestly with cross-validation and accept "no gain" when that is the result.

## Related folders

- [`../01-introduction/`](../01-introduction/) — what ensembles are and why they work (diversity).
- [`../02-voting/`](../02-voting/) — `VotingClassifier` / `VotingRegressor`: the fixed-weight "democracy" that stacking generalises.
