import argparse
import os
import numpy as np

from config import Config
from shapes import get_shape
from schedule import make_linear_schedule
from sampler import integrate_reverse_ode, integrate_reverse_sde
from score import score_gmm_time_marginal
from viz import save_frame, save_trails, frames_to_gif


def parse_args():
    parser = argparse.ArgumentParser(description="2D reverse-diffusion visualization")
    parser.add_argument("--shape", choices=["circle", "S"], default="circle")
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode")
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--beta_min", type=float, default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vis_every", type=int, default=None)
    parser.add_argument("--gif_fps", type=int, default=5, help="Frames per second for output GIFs.")
    parser.add_argument(
        "--gif_final_hold",
        type=int,
        default=20,
        help="Extra copies of the final frame to hold on the clean shape.",
    )
    parser.add_argument(
        "--make_reverse_gif",
        action="store_true",
        help="Also create a reverse-time GIF named 'reverse.gif'.",
    )
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
    if args.beta_min is not None:
        cfg.beta_min = args.beta_min
    if args.beta_max is not None:
        cfg.beta_max = args.beta_max
    if args.vis_every is not None:
        cfg.vis_every = args.vis_every
    cfg.shape = args.shape
    cfg.sampler = args.sampler

    rng = np.random.default_rng(cfg.seed)
    gmm = get_shape(cfg.shape)
    schedule = make_linear_schedule(cfg.steps, cfg.beta_min, cfg.beta_max)

    total_schedule_steps = len(schedule["beta"]) - 1
    print("First five (alpha, sigma) pairs:")
    for a, s in zip(schedule["alpha"][:5], schedule["sigma"][:5]):
        print(f"  alpha={a:.6f} sigma={s:.6f}")
    consistency = np.abs((1 - schedule["alpha"] ** 2) - schedule["sigma"] ** 2).max()
    print("abs((1-alpha^2)-sigma^2).max()", consistency)

    output_dir = os.path.join(cfg.output_dir, f"{cfg.shape}_{cfg.sampler}_seed{cfg.seed}")
    os.makedirs(output_dir, exist_ok=True)

    xT = rng.standard_normal((cfg.n_particles, 2))
    subset_count = min(cfg.trail_count, cfg.n_particles)
    subset_idx = rng.choice(cfg.n_particles, size=subset_count, replace=False)
    trails = []

    def callback(x, idx):
        trails.append(x[subset_idx].copy())
        if idx % cfg.vis_every == 0 or idx == total_schedule_steps:
            score_vals = score_gmm_time_marginal(
                x,
                gmm,
                alpha_t=float(schedule["alpha"][idx]),
                sigma_t=float(schedule["sigma"][idx]),
            )
            mean_score = float(np.mean(np.linalg.norm(score_vals, axis=1)))
            max_radius = float(np.max(np.linalg.norm(x, axis=1)))
            save_frame(
                x,
                gmm,
                step=idx,
                total_steps=total_schedule_steps,
                t_value=float(schedule["t"][idx]),
                output_dir=output_dir,
                sampler=cfg.sampler,
                shape=cfg.shape,
                score_mean=mean_score,
                max_radius=max_radius,
            )

    if cfg.sampler == "ode":
        samples = integrate_reverse_ode(xT, schedule, gmm, steps=None, callback=callback)
    else:
        samples = integrate_reverse_sde(xT, schedule, gmm, steps=None, seed=cfg.seed, callback=callback)

    trails_arr = np.stack(trails, axis=0)
    save_trails(trails_arr, output_dir, sampler=cfg.sampler, shape=cfg.shape)

    final_path = os.path.join(output_dir, "final.npy")
    np.save(final_path, samples)
    print(f"Saved final samples to {final_path}")
    print(f"Frames saved to {output_dir}")

    # Reverse diffusion GIF (noise -> clean) with a pause on the final clean shape.
    gif_path = frames_to_gif(
        output_dir=output_dir,
        gif_name="reverse.gif",
        fps=args.gif_fps,
        final_hold_frames=args.gif_final_hold,
        reverse=False,
    )
    if gif_path is not None:
        print(f"Saved reverse GIF to {gif_path}")
    else:
        print("No frames found for GIF creation.")

    # Optionally, create a forward diffusion GIF (clean -> noise).
    if args.make_reverse_gif:
        reverse_gif_path = frames_to_gif(
            output_dir=output_dir,
            gif_name="forward.gif",
            fps=args.gif_fps,
            final_hold_frames=0,
            reverse=True,
        )
        if reverse_gif_path is not None:
            print(f"Saved forward GIF to {reverse_gif_path}")


if __name__ == "__main__":
    main()
