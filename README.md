# Reverse Diffusion Simulation

> 📄 The full technical write-up will live in [`reports/final_report.pdf`](reports/final_report.pdf) (to be uploaded separately). This README summarizes the project and how to explore it today.

This repository provides a minimal, NumPy-based reverse diffusion simulator that mirrors the stochastic differential equation (SDE) math used in modern image-generation systems like Stable Diffusion and DALL·E. It integrates either the probability-flow ODE or the reverse-time SDE to transform Gaussian noise into structured samples drawn from a simple 2D Gaussian mixture model (GMM). The code is intentionally compact to expose the core algorithms behind diffusion models without the distractions of large neural networks.

## What the simulation does

* **Forward process** (conceptual): A clean “S” or ring-shaped GMM distribution is diffused into pure Gaussian noise using a variance-preserving (VP) SDE schedule. This mirrors how training data are noised during diffusion model training.
* **Reverse process** (implemented): Starting from standard Gaussian noise, the simulator integrates either the reverse SDE (Euler–Maruyama) or its deterministic probability-flow ODE (RK4). A handcrafted score function for the GMM replaces the learned neural network used in production models. Over many small steps, samples denoise back into the target shape, visualizing the generative trajectory seen in state-of-the-art text-to-image systems.
* **Outputs:** Scatter plots of the particle cloud are saved every `vis_every` steps, and the final particle positions are stored in `outputs/<run-name>/final.npy`. A GIF animation (generated from those frames) illustrates the reverse diffusion path.

## File and module overview

* `config.py`: Centralized simulation hyperparameters (time horizon, beta schedule, particle counts) plus utility to format run identifiers.
* `schedule.py`: Builds the VP noise schedule (`beta`, `alpha`, and `sigma` grids) that parameterize both the forward and reverse processes.
* `shapes.py`: Defines the target 2D Gaussian mixture (either a circle/ring or an "S" shape) and sampling helpers.
* `score.py`: Implements an exact score function for the time-marginal GMM, supplying the score network stand-in used during reverse integration.
* `sampler.py`: Houses the numerical integrators: a fourth-order Runge–Kutta solver for the probability-flow ODE and an Euler–Maruyama scheme for the reverse SDE.
* `viz.py`: Plotting utilities for scatter visualizations and animation frame saving.
* `run.py`: Command-line entry point that wires everything together—sampling initial noise, stepping the solver, and writing frames/results to `outputs/`.
* GIFs are generated locally from saved frames; none are committed to the repo to keep PRs binary-free.

## Running the simulator

Install the Python dependencies (NumPy, SciPy, Matplotlib, ImageIO):

```bash
pip install -r requirements.txt
```

Or install ad hoc:

```bash
pip install numpy scipy matplotlib imageio
```

Then launch a run (example: reverse SDE on the "S" shape):

```bash
python run.py --shape S --sampler sde --steps 1000 --n_particles 4000 --seed 0 --vis_every 25
```

Key flags:

* `--shape`: `circle` (ring) or `S` (S-curve Gaussian mixture).
* `--sampler`: `ode` for deterministic probability-flow integration or `sde` for stochastic reverse SDE.
* `--steps`: Number of integration steps (higher = smoother trajectories; also more frames).
* `--n_particles`: Number of particles simulated in parallel.
* `--vis_every`: Frame-saving cadence. Lower values yield smoother GIFs at the cost of more disk writes.

The run saves frames to `outputs/<shape>_<sampler>_seed<seed>/frame_*.png` and the final cloud to `outputs/<...>/final.npy`. You can convert the frames to a GIF via ImageIO, as demonstrated below.

## Numerical integration details

### Reverse SDE (Euler–Maruyama)
The variance-preserving SDE has drift `-0.5*beta_t*x - sigma_t^2 * score(x, t)` and diffusion `sigma_t`. Integrating backward from `t=T` to `t=0` uses Euler–Maruyama:

```
x_{k-1} = x_k + drift * dt + diffusion * sqrt(dt) * N(0, I)
```

This stochastic path mirrors the sampling procedure in latent diffusion models. The score term shapes the drift so particles flow toward high-density regions of the target distribution.

### Probability-flow ODE (RK4)
Replacing the stochastic term with a deterministic drift yields the probability-flow ODE. We integrate it with a 4th-order Runge–Kutta solver to reduce numerical error and visual jitter. This ODE shares marginals with the SDE—another standard trick used when generating images with diffusion models to avoid randomness or to enable exact likelihood computation.

### Relation to modern image generators
Diffusion-based text-to-image systems (e.g., Stable Diffusion, DALL·E) train neural networks to approximate the score function used here in closed form. At inference time, they integrate the same reverse SDE or its probability-flow ODE—just in a high-dimensional latent space instead of 2D—to turn noise into images. This toy simulator exposes that backbone math in a few hundred lines.

## Making your own GIF

After running `run.py`, stitch frames into an animation.

Quick option: use the provided helper script (keeps everything on-disk locally):

```bash
python make_gif.py --frames outputs/S_sde_seed0 --output outputs/S_sde_seed0/reverse_diffusion.gif --duration 0.08
```

> ℹ️ Generated GIFs are `.gitignore`d so you can build animations locally without adding binary files to pull requests.

Or assemble it inline in a one-off script:

```bash
python - <<'PY'
import imageio
from pathlib import Path
frames = sorted(Path('outputs/S_sde_seed0').glob('frame_*.png'))
images = [imageio.v2.imread(p) for p in frames]
imageio.mimsave('outputs/S_sde_seed0/reverse_diffusion.gif', images, duration=0.08)
PY
```

Move or copy the GIF wherever you need (e.g., `media/`) for sharing or embedding after generation.
