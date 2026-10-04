# Day 33 — Types of Gradient Descent

Batch (all rows per step), stochastic (one row per step) and mini-batch (a small group per step). The theory is in day 32's `01-gradient-descent-theory.ipynb`.

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | [`01-stochastic-gradient-descent-animation.ipynb`](01-stochastic-gradient-descent-animation.ipynb) | See the noisy stochastic path: line, cost and contour animations |
| 2 | [`02-EXTRA-batch-gradient-descent-from-scratch.ipynb`](02-EXTRA-batch-gradient-descent-from-scratch.ipynb) | **EXTRA.** Batch GD `GDRegressor` from scratch on the diabetes data vs `LinearRegression` |
| 3 | [`03-EXTRA-stochastic-gradient-descent-from-scratch.ipynb`](03-EXTRA-stochastic-gradient-descent-from-scratch.ipynb) | **EXTRA.** SGD from scratch, timing, and sklearn's `SGDRegressor` |
| 4 | [`04-EXTRA-mini-batch-gradient-descent-from-scratch.ipynb`](04-EXTRA-mini-batch-gradient-descent-from-scratch.ipynb) | **EXTRA.** Mini-batch GD from scratch, and `SGDRegressor.partial_fit` on mini-batches |

> Notebooks marked **EXTRA** implement the algorithm **from scratch**. They are optional, just for extra understanding: teach the numbered core notebooks first.

Other files: `stochastic_animation_*.gif` are saved from notebook 1; `mini_batch_contour_plot.gif` shows the mini-batch path on a contour plot.

## Links

Video Link:  
