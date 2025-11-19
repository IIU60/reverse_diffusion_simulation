from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 0
    steps: int = 1000
    total_time: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 20.0
    n_particles: int = 20_000
    shape: str = "circle"
    sampler: str = "ode"
    save_interval: int = 25
    output_dir: str = "outputs"
    vis_every: int = 25
    trail_count: int = 200
