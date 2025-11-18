import argparse
import os
import numpy as np

from config import Config
from shapes import get_shape
from schedule import make_linear_schedule
from score import make_score_fn
from sampler import probability_flow_ode, reverse_sde_sampler
from viz import save_scatter


def parse_args():
    parser = argparse.ArgumentParser(description="2D reverse-diffusion visualization")
    parser.add_argument("--shape", choices=["circle", "s"], default="circle")
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode")
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.n_particles is not None:
        cfg.n_particles = args.n_particles
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.shape = args.shape
    cfg.sampler = args.sampler

    rng = np.random.default_rng(cfg.seed)
    gmm = get_shape(cfg.shape)
    schedule = make_linear_schedule(cfg.steps, cfg.beta_min, cfg.beta_max)

    print("alpha[:3]", schedule["alpha"][:3])
    print("sigma[:3]", schedule["sigma"][:3])
    consistency = np.abs((1 - schedule["alpha"] ** 2) - schedule["sigma"] ** 2).max()
    print("abs((1-alpha^2)-sigma^2).max()", consistency)

    score_fn = make_score_fn(gmm.pi, gmm.mu, gmm.Sig, schedule)

    def callback(x, idx):
        if idx % cfg.save_interval == 0 or idx == 0:
            save_scatter(
                x,
                cfg.steps - idx,
                cfg.output_dir,
                cfg.shape,
                cfg.sampler,
                t_value=schedule["t"][idx],
            )

    if cfg.sampler == "ode":
        samples = probability_flow_ode(score_fn, schedule, cfg.n_particles, rng, callback=callback)
    else:
        samples = reverse_sde_sampler(rng, score_fn, schedule, cfg.n_particles, callback=callback)

    final_path = os.path.join(cfg.output_dir, "final.npy")
    np.save(final_path, samples)
    print(f"Saved final samples to {final_path}")


if __name__ == "__main__":
    main()
