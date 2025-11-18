from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rr_robot import RRRobot  # noqa: E402
from AStar import AStarPlanner  # noqa: E402


# -----------------------------------------------------------------------------
# 配置定义
# -----------------------------------------------------------------------------
@dataclass
class RobotConfig:
    L1: float = 2.0
    L2: float = 1.5
    theta1_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    theta2_range: Tuple[float, float] = (0.0, 2.0 * math.pi)


@dataclass
class DatasetConfig:
    num_samples: int = 1024
    resolution: int = 20
    min_obstacles: int = 1
    max_obstacles: int = 3
    obstacle_radius_min: float = 0.25
    obstacle_radius_max: float = 0.65
    max_attempts_per_sample: int = 200
    max_start_goal_tries: int = 80


# -----------------------------------------------------------------------------
# 数据集
# -----------------------------------------------------------------------------
class RRConfigSpaceDataset(Dataset):
    """通过几何求解 + A* 路径规划在线生成训练样本"""

    def __init__(
        self,
        robot_cfg: RobotConfig,
        dataset_cfg: DatasetConfig,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.dataset_cfg = dataset_cfg
        self.seed = seed
        self.verbose = verbose
        self.robot = RRRobot(
            robot_cfg.L1,
            robot_cfg.L2,
            robot_cfg.theta1_range,
            robot_cfg.theta2_range,
        )
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    # ------------------------------------------------------------------
    def _build_dataset(self) -> None:
        total = self.dataset_cfg.num_samples
        for idx in range(total):
            sample = self._build_single_sample(idx)
            self.samples.append(sample)
            if self.verbose:
                report_span = max(1, total // 10)
                if (idx + 1) % report_span == 0 or (idx + 1) == total:
                    print(f"[Dataset] 已生成 {idx + 1}/{total} 个样本", flush=True)

    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def _build_single_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng(idx)
        cfg = self.dataset_cfg

        for _ in range(cfg.max_attempts_per_sample):
            obstacles = self._random_obstacles(rng)
            config_space = self.robot.compute_configuration_space(
                obstacles, theta_resolution=cfg.resolution
            )
            if np.sum(config_space) < 4:
                continue

            planner = AStarPlanner(config_space, resolution=cfg.resolution)
            sg = self._sample_start_goal(config_space, planner, rng)
            if sg is None:
                continue

            _, _, path = sg
            return self._path_to_tensors(config_space, planner, path)

        raise RuntimeError("生成样本失败，无法找到可行的起终点配对")

    def _random_obstacles(
        self, rng: np.random.Generator
    ) -> List[Tuple[Tuple[float, float], float]]:
        cfg = self.dataset_cfg
        count = int(rng.integers(cfg.min_obstacles, cfg.max_obstacles + 1))
        reach = self.robot_cfg.L1 + self.robot_cfg.L2 - 0.3
        obstacles: List[Tuple[Tuple[float, float], float]] = []
        for _ in range(count):
            cx = float(rng.uniform(-reach, reach))
            cy = float(rng.uniform(-reach, reach))
            radius = float(rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max))
            obstacles.append(((cx, cy), radius))
        return obstacles

    def _sample_start_goal(
        self,
        config_space: np.ndarray,
        planner: AStarPlanner,
        rng: np.random.Generator,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]] | None:
        cfg = self.dataset_cfg
        free_positions = np.argwhere(config_space == 1)
        if free_positions.shape[0] < 2:
            return None

        for _ in range(cfg.max_start_goal_tries):
            idx = rng.choice(free_positions.shape[0], size=2, replace=False)
            start_idx = tuple(int(v) for v in free_positions[idx[0]])
            goal_idx = tuple(int(v) for v in free_positions[idx[1]])
            if start_idx == goal_idx:
                continue

            start_config = planner.grid_to_config(*start_idx)
            goal_config = planner.grid_to_config(*goal_idx)
            path = planner.plan_path(start_config, goal_config)
            if path and len(path) > 1:
                return start_config, goal_config, path

        return None

    def _path_to_tensors(
        self,
        config_space: np.ndarray,
        planner: AStarPlanner,
        path: List[Tuple[float, float]],
    ) -> Dict[str, torch.Tensor]:
        maze = config_space.astype(np.float32)
        traj_grid = np.zeros_like(maze, dtype=np.float32)
        for theta1, theta2 in path:
            gx, gy = planner.config_to_grid(theta1, theta2)
            traj_grid[gx, gy] = 1.0

        start_idx = planner.config_to_grid(*path[0])
        goal_idx = planner.config_to_grid(*path[-1])
        start_grid = np.zeros_like(maze, dtype=np.float32)
        goal_grid = np.zeros_like(maze, dtype=np.float32)
        start_grid[start_idx] = 1.0
        goal_grid[goal_idx] = 1.0

        return {
            "maze": torch.from_numpy(maze[None, ...]),
            "start": torch.from_numpy(start_grid[None, ...]),
            "goal": torch.from_numpy(goal_grid[None, ...]),
            "trajectory": torch.from_numpy(traj_grid[None, ...]),
        }

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.samples, path)
        print(f"[Dataset] 样本已保存到 {path}")


# -----------------------------------------------------------------------------
# 可视化
# -----------------------------------------------------------------------------
def visualize_dataset(
    dataset: RRConfigSpaceDataset,
    num_samples: int = 20,
    cols: int = 5,
    seed: int | None = None,
) -> None:
    if len(dataset) == 0:
        raise ValueError("数据集为空，无法可视化")

    rng = np.random.default_rng(seed)
    num_samples = min(num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)

    maze_cmap = ListedColormap(["#8b1a1a", "#fef9f4"])

    for ax, idx in zip(axes.flatten(), indices):
        sample = dataset[int(idx)]
        maze = sample["maze"][0].numpy()
        traj = sample["trajectory"][0].numpy()
        start = sample["start"][0].numpy()
        goal = sample["goal"][0].numpy()

        ax.imshow(maze, cmap=maze_cmap, vmin=0, vmax=1, interpolation="nearest")

        traj_mask = np.ma.masked_where(traj == 0, traj)
        ax.imshow(
            traj_mask,
            cmap="Blues",
            alpha=0.95,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        if np.any(traj > 0):
            ax.contour(
                traj,
                levels=[0.5],
                colors="#1f77b4",
                linewidths=1.6,
                linestyles="solid",
            )

        def _plot_marker(mask: np.ndarray, marker: str, color: str) -> None:
            coords = np.argwhere(mask > 0.5)
            if coords.size == 0:
                return
            ax.scatter(
                coords[:, 1],
                coords[:, 0],
                marker=marker,
                c=color,
                s=80,
                edgecolors="black",
                linewidths=0.6,
            )

        _plot_marker(start, "o", "#2ca02c")
        _plot_marker(goal, "X", "#d62728")

        ax.set_title(f"idx={idx}")
        ax.axis("off")

    # 处理多余子图
    for ax in axes.flatten()[num_samples:]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor="#8b1a1a", label="obstacle / infeasible"),
        Patch(facecolor="#fef9f4", label="feasible region"),
        Patch(facecolor="#4f9ae5", alpha=0.8, label="planned trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markersize=8,
            markerfacecolor="#2ca02c",
            label="start point",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="black",
            markersize=8,
            markerfacecolor="#d62728",
            label="goal point",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("RR configuration space dataset visualization", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RR 机器人配置空间数据集生成脚本")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--min-obstacles", type=int, default=1)
    parser.add_argument("--max-obstacles", type=int, default=3)
    parser.add_argument("--obstacle-radius-min", type=float, default=0.25)
    parser.add_argument("--obstacle-radius-max", type=float, default=0.65)
    parser.add_argument("--max-attempts-per-sample", type=int, default=200)
    parser.add_argument("--max-start-goal-tries", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rr_dataset.pt"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-samples", type=int, default=20)
    parser.add_argument("--vis-cols", type=int, default=5)
    parser.add_argument("--no-save", action="store_true", help="不落盘，只可视化")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_cfg = DatasetConfig(
        num_samples=args.num_samples,
        resolution=args.resolution,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        obstacle_radius_min=args.obstacle_radius_min,
        obstacle_radius_max=args.obstacle_radius_max,
        max_attempts_per_sample=args.max_attempts_per_sample,
        max_start_goal_tries=args.max_start_goal_tries,
    )
    robot_cfg = RobotConfig()

    dataset = RRConfigSpaceDataset(robot_cfg, dataset_cfg, seed=args.seed, verbose=not args.quiet)

    if not args.no_save and args.output:
        dataset.save(args.output)

    if args.visualize:
        visualize_dataset(dataset, num_samples=args.vis_samples, cols=args.vis_cols)


if __name__ == "__main__":
    main()
