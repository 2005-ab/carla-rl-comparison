"""Train genuine Stable-Baselines3 policies and evaluate on held-out local seeds."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from common.local_env import LocalDrivingEnv, baseline_action, SCENARIOS
from common.evaluation import evaluate_policy, aggregate_results

ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3}
ROOT = Path(__file__).resolve().parents[1]


class Progress(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 5000 == 0:
            print(f"  {self.model.__class__.__name__}: {self.num_timesteps:,} steps", flush=True)
        return True


def make_model(algorithm, env, seed):
    common = dict(policy="MlpPolicy", env=env, seed=seed, device="cpu", verbose=0,
                  learning_rate=3e-4, gamma=.99, policy_kwargs=dict(net_arch=[64, 64]))
    if algorithm == "ppo":
        return PPO(**common, n_steps=1024, batch_size=128, n_epochs=10, ent_coef=.005)
    extra = dict(buffer_size=100_000, learning_starts=1000, batch_size=128, train_freq=1, gradient_steps=1)
    if algorithm == "sac":
        return SAC(**common, **extra, ent_coef="auto")
    return TD3(**common, **extra, policy_delay=2, target_policy_noise=.2,
               action_noise=NormalActionNoise(np.zeros(3), .15*np.ones(3)))


def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3.common.monitor import load_results
    combined = aggregate_results(str(directory))
    if combined.empty:
        return
    lines = ["# Measured local simulator results", "",
             "These measurements come from actual trained policies in the local 2D environment, not CARLA.", "",
             "| Policy | Return | Goal success | Collision rate | Route completion | Lane error |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, row in combined.iterrows():
        lines.append(f"| {row.algorithm} | {row.mean_reward:.1f} | {row.goal_success_rate_pct:.1f}% | {row.collision_rate_pct:.1f}% | {row.route_completion_pct_mean:.1f}% | {row.mean_lateral_deviation_m:.3f} m |")
    lines += ["", "BASELINE is a hand-coded reference, not a learned policy.", "",
              "The test seeds, scenario, inference mode, and source model paths are in evaluation_protocol.json. Training budgets are recorded with the models under models/local/.",
              "This is a short, single training-seed experiment. It does not establish convergence or a general algorithm ranking.", "",
              "![Held-out evaluation](comparison.png)"]
    if directory.name == "two_lane":
        lines += ["", "This is the two-lane challenge. When challenge weights are absent, evaluation uses original-scenario models without retraining; see source paths in the protocol. Old benchmark results are kept separately."]
    colors = {"PPO": "#32c6cb", "SAC": "#f0b54c", "TD3": "#8d9cff", "BASELINE": "#8abf95"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.flat, [("mean_reward", "Episode return"), ("goal_success_rate_pct", "Goal success (%)"), ("collision_rate_pct", "Collision rate (%) - lower is better"), ("route_completion_pct_mean", "Route completed (%)")]):
        ax.bar(combined.algorithm, combined[metric], color=[colors.get(name, "#999999") for name in combined.algorithm])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Local 2D driving benchmark | held-out seeds | not CARLA results")
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for algorithm in ALGORITHMS:
        folder = directory / algorithm
        if not folder.exists() or not any(folder.glob("*.monitor.csv")):
            continue
        df = load_results(str(folder))
        if len(df):
            ax.plot(df.l.cumsum(), df.r.rolling(20, min_periods=1).mean(), label=algorithm.upper(), color=colors[algorithm.upper()])
    ax.set(xlabel="Environment transitions", ylabel="Return (20 episode moving mean)", title="Actual local training history")
    if ax.lines:
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory / "learning_curves.png", dpi=160)
        lines += ["", "![Actual training history](learning_curves.png)"]
    plt.close(fig)
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["all", *ALGORITHMS], default="all")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("steps and episodes must be positive")
    torch.set_num_threads(1)
    output = ROOT / "results" / "local"
    if args.scenario == "two_lane":
        output /= "two_lane"
    output.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "local"
    if args.scenario == "two_lane":
        model_dir /= "two_lane"
    model_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1001, 1001+args.episodes))
    algorithms = list(ALGORITHMS) if args.algo == "all" else [args.algo]
    model_sources = {}
    for algorithm in algorithms:
        log_dir = output / algorithm
        log_dir.mkdir(exist_ok=True)
        env = LocalDrivingEnv(scenario=args.scenario)
        if not args.evaluate_only:
            env = Monitor(env, str(log_dir / "train"))
        started = time.perf_counter()
        try:
            if args.evaluate_only:
                source = model_dir / f"{algorithm}.zip"
                if not source.exists() and args.scenario == "two_lane":
                    source = ROOT / "models" / "local" / f"{algorithm}.zip"
                    print(f"{algorithm.upper()}: testing original model on unseen two-lane challenge", flush=True)
                model = ALGORITHMS[algorithm].load(source, device="cpu")
                duration = None
            else:
                print(f"Training {algorithm.upper()} for {args.steps:,} requested steps", flush=True)
                model = make_model(algorithm, env, args.seed)
                model.learn(total_timesteps=args.steps, callback=Progress())
                duration = time.perf_counter()-started
                model.save(model_dir / algorithm)
                source = model_dir / f"{algorithm}.zip"
                metadata = dict(algorithm=algorithm, backend="local_2d", requested_steps=args.steps,
                                actual_steps=model.num_timesteps, seed=args.seed, scenario=args.scenario,
                                training_seconds=duration, observation_dim=10, action_dim=3,
                                config=env.unwrapped.config.to_dict(), hyperparameters={
                                    "learning_rate": .0003, "gamma": .99, "net_arch": [64, 64],
                                    "batch_size": 128, "n_steps": 1024 if algorithm == "ppo" else None,
                                    "learning_starts": 1000 if algorithm != "ppo" else None},
                                versions={"python":sys.version,"torch":torch.__version__,
                                          "stable_baselines3": stable_baselines3.__version__,
                                          "gymnasium": gymnasium.__version__})
                (model_dir / f"{algorithm}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            model_sources[algorithm] = str(source.relative_to(ROOT))
            evaluation_env = LocalDrivingEnv(scenario=args.scenario)
            try:
                summary, _ = evaluate_policy(model, evaluation_env, algorithm, args.episodes,
                    seeds=seeds, results_dir=str(output), training_wall_clock_sec=duration)
                print(json.dumps(summary.to_dict(), indent=2), flush=True)
            finally:
                evaluation_env.close()
        finally:
            env.close()
    baseline_env = LocalDrivingEnv(scenario=args.scenario)
    try:
        evaluate_policy(lambda obs: baseline_action(baseline_env), baseline_env, "BASELINE", args.episodes,
                        seeds=seeds, results_dir=str(output))
    finally:
        baseline_env.close()
    plot_results(output)
    (output / "evaluation_protocol.json").write_text(json.dumps(dict(
        backend="local_2d", scenario=args.scenario, seeds=seeds, deterministic=True,
        model_sources=model_sources,
        note="Single training seed; preliminary local results, not CARLA validation."), indent=2), encoding="utf-8")
    print(f"Model directory: {model_dir}\nSaved results: {output}", flush=True)


if __name__ == "__main__":
    main()
