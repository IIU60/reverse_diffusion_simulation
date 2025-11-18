# Reverse Diffusion Simulation

A minimal NumPy + Matplotlib scaffold for visualizing 2D reverse-diffusion with the VP (Ornstein–Uhlenbeck–like) SDE.

## Usage

```bash
python run.py --shape circle --sampler ode
```

Arguments:
- `--shape`: `circle` or `s` for the underlying Gaussian mixture shape.
- `--sampler`: `ode` for probability-flow RK4 integration or `sde` for reverse SDE Euler–Maruyama.
- `--n_particles`: number of particles to sample (default 512).
- `--steps`: number of integration steps (default 500).
- `--seed`: RNG seed (default 0).

The script prints early schedule values, checks `(1-α^2)≈σ^2`, saves scatter PNGs every 25 steps in `outputs/`, and writes the final samples to `outputs/final.npy`.
