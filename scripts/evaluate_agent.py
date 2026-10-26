#!/usr/bin/env python3
"""
scripts/evaluate_agent.py

Loads a trained Stable-Baselines3 model (SAC, TD3, or PPO), evaluates it on the
shared CarlaDrivingEnv using the standardized evaluation protocol, and writes
per-episode + summary CSVs to the results directory.

Usage:
    python scripts/evaluate_agent.py --model models/sac_carla.zip --algo SAC
        [--host localhost] [--port 2000] [--episodes 10] [--results-dir results]
"""

import argparse
import os
import sys
import time

# Make the workspace root importable when running this script directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import EnvConfig, EvaluationConfig, SimulatorConfig
from common.carla_env import CarlaDrivingEnv, CarlaConnectionError
from common.evaluation import evaluate_policy, aggregate_results


_ALGO_REGISTRY = {
    "sac": ("stable_baselines3", "SAC"),
    "td3": ("stable_baselines3", "TD3"),
    "ppo": ("stable_baselines3", "PPO"),
}


def load_model(algo: str, model_path: str):
    """Load an SB3 model by algorithm name with a clear error on mismatch."""
    key = algo.lower()
    if key not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algo}'. Expected one of {sorted(_ALGO_REGISTRY)}.")
    try:
        import stable_baselines3 as sb3
    except ImportError as e:
        raise RuntimeError(
            "stable_baselines3 is not installed. Install with: pip install stable-baselines3"
        ) from e

    module_name, class_name = _ALGO_REGISTRY[key]
    cls = getattr(sb3, class_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return cls.load(model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained CARLA RL agent")
    parser.add_argument("--model", required=True, help="Path to the saved SB3 model (.zip)")
    parser.add_argument("--algo", required=True, help="Algorithm: SAC, TD3, or PPO")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--training-seconds", type=float, default=None,
                        help="Optional training wall-clock time to record in the summary")
    args = parser.parse_args()

    try:
        model = load_model(args.algo, args.model)
    except Exception as e:
        print(f"ERROR: could not load model: {e}", file=sys.stderr)
        return 2

    config = EnvConfig()
    config.simulator = SimulatorConfig(host=args.host, port=args.port)
    eval_cfg = EvaluationConfig(n_eval_episodes=args.episodes, deterministic=args.deterministic)

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    try:
        env = CarlaDrivingEnv(config=config, connect_on_init=True)
    except CarlaConnectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    try:
        print(f"Evaluating {args.algo.upper()} for {args.episodes} episodes ...")
        t0 = time.time()
        summary, df = evaluate_policy(
            model=model,
            env=env,
            algorithm_name=args.algo,
            n_eval_episodes=eval_cfg.n_eval_episodes,
            deterministic=eval_cfg.deterministic,
            seeds=list(eval_cfg.eval_seeds),
            results_dir=args.results_dir,
            training_wall_clock_sec=args.training_seconds,
        )
        eval_time = time.time() - t0
        print(f"Evaluation finished in {eval_time:.1f}s")
        print("Summary:")
        for key, value in summary.to_dict().items():
            print(f"  {key}: {value}")
        print(f"Per-episode CSV written under: {args.results_dir}")

        # Combine all available summaries for direct cross-algorithm comparison.
        try:
            aggregate_results(args.results_dir)
        except Exception as e:
            print(f"WARNING: could not aggregate results: {e}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"ERROR: evaluation failed: {e}", file=sys.stderr)
        return 4
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
