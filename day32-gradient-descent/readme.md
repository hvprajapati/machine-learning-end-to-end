# Day 32 — Gradient Descent

Find the best line by walking downhill on the cost function, one small step at a time.

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | [`01-gradient-descent-theory.ipynb`](01-gradient-descent-theory.ipynb) | What gradient descent is, intuition, the gradient, update rule, learning rate, types (overview), updates for linear regression, convergence |
| 2 | [`02-gradient-descent-step-by-step-intercept-only.ipynb`](02-gradient-descent-step-by-step-intercept-only.ipynb) | 4 data points, slope fixed: update the intercept `b` by hand, iteration by iteration, then in a loop |
| 3 | [`03-gradient-descent-animation-intercept-only.ipynb`](03-gradient-descent-animation-intercept-only.ipynb) | Animations of `b` converging: regression line, cost vs epochs, `b` vs epochs, cost vs `b` |
| 4 | [`04-gradient-descent-animation-slope-and-intercept.ipynb`](04-gradient-descent-animation-slope-and-intercept.ipynb) | Animations of `m` and `b` learned together |
| 5 | [`05-gradient-descent-3d-cost-surface-and-contour.ipynb`](05-gradient-descent-3d-cost-surface-and-contour.ipynb) | The cost surface in 3-D and as a contour plot, with the gradient descent path |
| 6 | [`06-EXTRA-gradient-descent-from-scratch.ipynb`](06-EXTRA-gradient-descent-from-scratch.ipynb) | **EXTRA.** A `GDRegressor` class that learns `m` and `b`, compared with `LinearRegression` |

> Notebooks marked **EXTRA** implement the algorithm **from scratch**. They are optional, just for extra understanding: teach the numbered core notebooks first.

Other files: `animation*.gif`, `cost_function*.html` and `gradient-descent.png` are pictures used by or saved from these notebooks.  
Next: [`../day33-types-of-gradient-descent/`](../day33-types-of-gradient-descent/)

## Links

Video Link  
