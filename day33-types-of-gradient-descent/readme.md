# Day 33 — Types of Gradient Descent

Video Link:

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | `01-batch-gradient-descent.ipynb` | Batch GD: the whole training set for every update (diabetes data), `GDRegressor` from scratch vs `LinearRegression` |
| 2 | `02-stochastic-gradient-descent-from-scratch.ipynb` | Stochastic GD: one row per update, from scratch, timing, and sklearn's `SGDRegressor` |
| 3 | `03-stochastic-gradient-descent-animation.ipynb` | Animations of the noisy SGD path (line, cost and contour plots) |
| 4 | `04-mini-batch-gradient-descent-from-scratch.ipynb` | Mini-batch GD from scratch, and with `SGDRegressor.partial_fit` on mini-batches |

Other files: `stochastic_animation_*.gif` are animations saved from notebook 3; `mini_batch_contour_plot.gif` shows the mini-batch path on a contour plot (no notebook here creates it).

Previous: `../day32-gradient-descent/` (its theory notebook also introduces batch / stochastic / mini-batch).
