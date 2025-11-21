"""Utility to assemble reverse-diffusion frames into a GIF.

Example:
    python make_gif.py --frames outputs/S_sde_seed0 --output media/reverse_diffusion.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio


def assemble_gif(
    frame_dir: Path, pattern: str = "frame_*.png", duration: float = 0.08, output: Path | None = None
) -> Path:
    """Load ordered frames from ``frame_dir`` and write a GIF.

    Args:
        frame_dir: Directory containing saved frames from ``run.py``.
        pattern: Glob pattern to match frames (default ``frame_*.png``).
        duration: Duration per frame in seconds.
        output: Destination path for the GIF. Defaults to ``frame_dir / "reverse_diffusion.gif"``.

    Returns:
        Path to the written GIF.
    """

    frames: List[Path] = sorted(frame_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching '{pattern}' found in {frame_dir}")

    output = output or frame_dir / "reverse_diffusion.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output, images, duration=duration)
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble reverse-diffusion frames into a GIF")
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="Directory containing frame PNGs from run.py (e.g., outputs/S_sde_seed0)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.png",
        help="Glob pattern for frames (default: frame_*.png)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.08,
        help="Seconds per frame in the GIF (default: 0.08)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path (default: <frames>/reverse_diffusion.gif)",
    )

    args = parser.parse_args(argv)
    gif_path = assemble_gif(args.frames, args.pattern, args.duration, args.output)
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
