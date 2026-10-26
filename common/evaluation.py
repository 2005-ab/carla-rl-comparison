"""
common/evaluation.py

Standardized evaluation protocol for CARLA RL agents (SAC, TD3, PPO).
Ensures strictly identical evaluation scenarios, random seeds, and metric definitions.

Metric Denominators and Definitions:
  - Total Episodes (N): Total number of evaluation episodes executed.
  - Goal Success Rate (%): 100.0 * (Episodes where goal was reached / N)
  - Collision Rate (%): 100.0 * (Episodes where >= 1 collision occurred / N)
  - Mean Collisions Per Episode: Total collision events detected across all steps / N
  - Route Completion (%): Average of episode final route completion percentages
  - Mean Speed (km/h): Average vehicle speed across all evaluation steps
  - Mean Lateral Deviation (m): Average absolute distance from lane centerline across all steps
"""

from dataclasses import dataclass, asdict
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from common.config import CarlaConfig, EvaluationConfig


@dataclass
class EpisodeResult:
    """Individual evaluation episode outcome."""
    episode: int
    seed: int
    total_reward: float
    route_completion_pct: float
    goal_reached: bool
    collision: bool
    num_collisions: int
    duration_steps: int
    duration_seconds: float
    mean_speed_kmh: float
    mean_lateral_deviation_m: float


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across all test episodes."""
    algorithm: str
    total_episodes: int
    mean_reward: float
    std_reward: float
    route_completion_pct_mean: float
    route_completion_pct_std: float
    goal_success_rate_pct: float
    collision_rate_pct: float
    mean_collisions_per_episode: float
    mean_duration_steps: float
    mean_duration_seconds: float
    mean_speed_kmh: float
    mean_lateral_deviation_m: float
    evaluation_wall_clock_sec: float
    training_wall_clock_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_policy(
    model: Any,
    env: Any,
    algorithm_name: str,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    seeds: Optional[List[int]] = None,
    results_dir: str = "results",
    training_wall_clock_sec: Optional[float] = None,
) -> Tuple[EvaluationSummary, pd.DataFrame]:
    """
    Evaluates a trained RL agent using the standardized benchmark protocol.

    Parameters
    ----------
    model : BaseAlgorithm (Stable-Baselines3 model)
        The trained policy (SAC, TD3, or PPO).
    env : gym.Env
        The evaluation driving environment.
    algorithm_name : str
        Identifier ('SAC', 'TD3', 'PPO').
    n_eval_episodes : int
        Number of distinct test episodes.
    deterministic : bool
        If True, use deterministic action selection.
    seeds : Optional[List[int]]
        Fixed seeds for evaluation runs.
    results_dir : str
        Directory to store evaluation CSV reports.
    training_wall_clock_sec : Optional[float]
        Training duration if recorded.

    Returns
    -------
    Tuple[EvaluationSummary, pd.DataFrame]
        Summary dataclass and detailed per-episode DataFrame.
    """
    eval_seeds = seeds or [1000 + i for i in range(n_eval_episodes)]
    if len(eval_seeds) < n_eval_episodes:
        eval_seeds.extend([2000 + i for i in range(n_eval_episodes - len(eval_seeds))])

    episode_results: List[EpisodeResult] = []
    eval_start_time = time.time()

    for ep_idx in range(n_eval_episodes):
        ep_seed = eval_seeds[ep_idx]
        obs, info = env.reset(seed=ep_seed)
        terminated = False
        truncated = False

        total_reward = 0.0
        step_speeds: List[float] = []
        step_lat_devs: List[float] = []
        num_collisions = 0
        step_count = 0
        last_info = info

        while not (terminated or truncated):
            # Model prediction (strictly evaluation mode, no gradients or updates)
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Fallback for callable policy or baseline function
                action = model(obs)

            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            step_speeds.append(last_info.get("speed_kmh", 0.0))
            step_lat_devs.append(abs(last_info.get("lateral_deviation_m", 0.0)))
            if last_info.get("collision", False):
                num_collisions += 1

        duration_sec = step_count * env.config.simulator.fixed_delta_seconds
        has_collision = num_collisions > 0 or last_info.get("collision", False)
        goal_reached = bool(last_info.get("goal_reached", False))
        route_comp = float(last_info.get("route_completion_pct", 0.0))
        mean_speed = float(np.mean(step_speeds)) if step_speeds else 0.0
        mean_lat = float(np.mean(step_lat_devs)) if step_lat_devs else 0.0

        ep_res = EpisodeResult(
            episode=ep_idx + 1,
            seed=ep_seed,
            total_reward=round(total_reward, 3),
            route_completion_pct=round(route_comp, 2),
            goal_reached=goal_reached,
            collision=has_collision,
            num_collisions=num_collisions,
            duration_steps=step_count,
            duration_seconds=round(duration_sec, 2),
            mean_speed_kmh=round(mean_speed, 2),
            mean_lateral_deviation_m=round(mean_lat, 3),
        )
        episode_results.append(ep_res)

    eval_total_time = time.time() - eval_start_time

    # Construct DataFrame
    df_episodes = pd.DataFrame([asdict(r) for r in episode_results])

    # Compute explicit aggregate metrics
    n = len(episode_results)
    rewards = [r.total_reward for r in episode_results]
    route_comps = [r.route_completion_pct for r in episode_results]
    goal_successes = sum(1 for r in episode_results if r.goal_reached)
    collision_episodes = sum(1 for r in episode_results if r.collision)
    total_collision_count = sum(r.num_collisions for r in episode_results)

    summary = EvaluationSummary(
        algorithm=algorithm_name.upper(),
        total_episodes=n,
        mean_reward=round(float(np.mean(rewards)), 3),
        std_reward=round(float(np.std(rewards)), 3),
        route_completion_pct_mean=round(float(np.mean(route_comps)), 2),
        route_completion_pct_std=round(float(np.std(route_comps)), 2),
        goal_success_rate_pct=round(100.0 * goal_successes / max(1, n), 2),
        collision_rate_pct=round(100.0 * collision_episodes / max(1, n), 2),
        mean_collisions_per_episode=round(total_collision_count / max(1, n), 2),
        mean_duration_steps=round(float(np.mean([r.duration_steps for r in episode_results])), 1),
        mean_duration_seconds=round(float(np.mean([r.duration_seconds for r in episode_results])), 2),
        mean_speed_kmh=round(float(np.mean([r.mean_speed_kmh for r in episode_results])), 2),
        mean_lateral_deviation_m=round(float(np.mean([r.mean_lateral_deviation_m for r in episode_results])), 3),
        evaluation_wall_clock_sec=round(eval_total_time, 2),
        training_wall_clock_sec=round(training_wall_clock_sec, 2) if training_wall_clock_sec else None,
    )

    # Save CSVs
    os.makedirs(results_dir, exist_ok=True)
    algo_key = algorithm_name.lower().replace(" ", "_")

    episodes_csv_path = os.path.join(results_dir, f"{algo_key}_episodes.csv")
    df_episodes.to_csv(episodes_csv_path, index=False)

    summary_csv_path = os.path.join(results_dir, f"{algo_key}_summary.csv")
    df_summary = pd.DataFrame([summary.to_dict()])
    df_summary.to_csv(summary_csv_path, index=False)

    return summary, df_episodes


def aggregate_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Scans results_dir for all algorithm summary CSVs and combines them into a single table.
    """
    summary_files = [
        f for f in os.listdir(results_dir) if f.endswith("_summary.csv") and f != "combined_summary.csv"
    ]
    dfs = []
    for f in summary_files:
        p = os.path.join(results_dir, f)
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(results_dir, "combined_summary.csv")
    combined_df.to_csv(combined_path, index=False)
    return combined_df


def plot_algorithm_comparison(
    combined_df: pd.DataFrame,
    output_image_path: str = "results/algorithm_comparison.png",
) -> Optional[str]:
    """
    Generates comparison bar charts across evaluated algorithms.
    """
    if combined_df.empty or "algorithm" not in combined_df.columns:
        return None

    try:
        import matplotlib.pyplot as plt

        metrics = [
            ("mean_reward", "Mean Episode Reward"),
            ("route_completion_pct_mean", "Route Completion (%)"),
            ("goal_success_rate_pct", "Goal Success Rate (%)"),
            ("collision_rate_pct", "Collision Rate (%)"),
            ("mean_speed_kmh", "Mean Speed (km/h)"),
            ("mean_lateral_deviation_m", "Mean Lateral Deviation (m)"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        algos = combined_df["algorithm"].tolist()
        colors = ["#4C72B0", "#55A868", "#C44E52"][:len(algos)]

        for idx, (col, title) in enumerate(metrics):
            ax = axes[idx]
            if col in combined_df.columns:
                vals = combined_df[col].tolist()
                bars = ax.bar(algos, vals, color=colors, alpha=0.85, edgecolor="black")
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                # Value labels on top of bars
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        yval + (0.02 * abs(yval) if yval >= 0 else -0.05 * abs(yval)),
                        f"{yval:.2f}",
                        ha="center",
                        va="bottom" if yval >= 0 else "top",
                        fontsize=10,
                    )

        plt.suptitle("CARLA RL Benchmark: SAC vs TD3 vs PPO", fontsize=16, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        return output_image_path
    except Exception as e:
        print(f"Warning: Could not generate comparison plot: {e}")
        return None
