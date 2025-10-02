# ilqr

Python implementation of **AL-iLQR** for Trajectory optimization 
For details, see the original [conference paper](http://roboticexplorationlab.org/papers/altro-iros.pdf) or [Julia Altro](https://github.com/RoboticExplorationLab/Altro.jl) 

## Requirements

- Python **3.10 – 3.12**
- [`uv`](https://docs.astral.sh/uv/) (fast package/env manager)

> Install `uv` (one-liner):
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
>
> Or on macOS (Homebrew):
>
> ```bash
> brew install uv
> ```

---

## Install (one command)

From the project root (where `pyproject.toml` lives):

```bash
uv sync
```

## Run 

```bash
# run ilqr with car model
uv run python -m examples.car

# meshcat visualization
uv run python -m test.test_car
```
<div align="center">
  <img src="images/output_car_.gif" alt="main" width="50%">
</div>


## TODO
- [x] AL-ilqr with basic numpy
- [ ] AL-ilqr with JAX
- [ ] Comparison with Julia
