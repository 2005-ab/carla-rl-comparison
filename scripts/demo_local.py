"""Interactive presentation: choose a learned policy, reference controller, or manual driving."""
import argparse
import csv
from copy import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.local_env import LocalDrivingEnv, baseline_action, SCENARIOS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["baseline", "manual", "ppo", "sac", "td3"], default="baseline")
    parser.add_argument("--scenario", choices=SCENARIOS, default="curve")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames (smoke tests)")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--view", choices=["3d", "2d"], default="3d")
    args = parser.parse_args()
    import pygame
    from common.local_renderer import DrivingRenderer
    from common.local_renderer3d import DrivingRenderer3D
    env = LocalDrivingEnv(scenario=args.scenario)
    view_mode = args.view
    renderer = DrivingRenderer3D("human") if view_mode == "3d" else DrivingRenderer("human")
    model_cache = {}
    model_origins = {}
    policy, scenario, seed = args.policy, args.scenario, args.seed
    scenarios = list(SCENARIOS)
    notice = ""
    def choose(name):
        nonlocal policy, notice
        if name in ("ppo", "sac", "td3"):
            path = ROOT / "models" / "local" / f"{name}.zip"
            challenge_path = ROOT / "models" / "local" / "two_lane" / f"{name}.zip"
            if scenario == "two_lane" and challenge_path.exists():
                path = challenge_path
            if not path.exists():
                notice = f"{name.upper()} missing: run scripts/train_local.py first"
                print(notice)
                return
            if model_origins.get(name) != path:
                import torch
                torch.set_num_threads(1)
                from stable_baselines3 import PPO, SAC, TD3
                model_cache[name] = {"ppo": PPO, "sac": SAC, "td3": TD3}[name].load(path, device="cpu")
                model_origins[name] = path
        policy, notice = name, ""
    policy = "baseline"
    choose(args.policy)
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    def refresh_comparison():
        folder = ROOT / "results" / "local"
        if env.is_two_lane:
            folder /= "two_lane"
        summary = folder / "combined_summary.csv"
        renderer.comparison = []
        if summary.exists():
            with summary.open(newline="", encoding="utf-8") as f:
                renderer.comparison = list(csv.DictReader(f))
    refresh_comparison()
    clock = pygame.time.Clock()
    running, paused, frame, end_time, multiplier = True, False, 0, 0.0, 1
    accumulator = 0.0
    def pose():
        return (env.x, env.y, env.yaw, env.speed, [car.copy() for car in env.traffic])
    previous_pose = pose()
    try:
        while running:
            # Fixed 10 Hz decisions, independent 60 Hz rendering. Frame-limited smoke
            # runs use deterministic time and run uncapped for fast screenshot capture.
            elapsed = 1/60 if args.frames else min(clock.tick(60)/1000, .25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_c and hasattr(renderer, "cycle_camera"):
                        renderer.cycle_camera()
                    if event.key == pygame.K_v:
                        comparison = renderer.comparison
                        renderer.close()
                        view_mode = "2d" if view_mode == "3d" else "3d"
                        renderer = DrivingRenderer3D("human") if view_mode == "3d" else DrivingRenderer("human")
                        renderer.comparison = comparison
                    if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        multiplier = min(8, multiplier*2)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        multiplier = max(1, multiplier//2)
                    mapping = {pygame.K_1: "ppo", pygame.K_2: "sac", pygame.K_3: "td3", pygame.K_b: "baseline", pygame.K_m: "manual"}
                    if event.key in mapping:
                        choose(mapping[event.key])
                        obs, _ = env.reset(seed=seed, options={"scenario": scenario})
                        end_time, accumulator = 0.0, 0.0
                        previous_pose = pose()
                    if event.key in (pygame.K_r, pygame.K_t, pygame.K_n):
                        if event.key == pygame.K_t:
                            scenario = scenarios[(scenarios.index(scenario)+1)%len(scenarios)]
                            choose(policy)
                        if event.key == pygame.K_n:
                            seed += 1
                        obs, _ = env.reset(seed=seed, options={"scenario": scenario})
                        end_time, accumulator = 0.0, 0.0
                        previous_pose = pose()
                        refresh_comparison()
            if not running:
                break
            if not paused:
                if env.done:
                    end_time += elapsed
                    if end_time > 2:
                        obs, _ = env.reset(seed=seed, options={"scenario": scenario})
                        end_time, accumulator = 0.0, 0.0
                        previous_pose = pose()
                else:
                    accumulator += elapsed*multiplier
                    while accumulator >= env.config.simulator.fixed_delta_seconds:
                        previous_pose = pose()
                        accumulator -= env.config.simulator.fixed_delta_seconds
                        if policy == "manual":
                            keys = pygame.key.get_pressed()
                            action = np.array([float(keys[pygame.K_RIGHT] or keys[pygame.K_d])-float(keys[pygame.K_LEFT] or keys[pygame.K_a]),
                                               float(keys[pygame.K_UP] or keys[pygame.K_w]), float(keys[pygame.K_DOWN] or keys[pygame.K_s])], dtype=np.float32)
                        elif policy == "baseline":
                            action = baseline_action(env)
                        else:
                            action, _ = model_cache[policy].predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
            description = "hand-coded reference" if policy == "baseline" else "WASD / arrow keys" if policy == "manual" else "trained neural policy"
            if env.is_two_lane and policy in model_origins:
                description = "two-lane trained policy" if model_origins[policy].parent.name == "two_lane" else "original model / new challenge"
            renderer.label = notice or f"{policy.upper()} / {description}   |   {multiplier}x   |   seed {seed}"
            renderer.paused = paused
            # Interpolate display-only state. Rendering never changes policy inputs,
            # rewards, collisions, the environment's RNG or recorded benchmark results.
            display_env = env
            if not (paused or env.done):
                alpha = accumulator/env.config.simulator.fixed_delta_seconds
                display_env = copy(env)
                for name, old in zip(("x", "y", "yaw", "speed"), previous_pose[:4]):
                    setattr(display_env, name, old+(getattr(env,name)-old)*alpha)
                display_env.traffic = [[old+(current-old)*alpha for old,current in zip(before,after)]
                                       for before,after in zip(previous_pose[4],env.traffic)]
            renderer.draw(display_env)
            frame += 1
            if args.frames and frame >= args.frames:
                running = False
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(renderer.surface, str(args.screenshot))
    finally:
        env.close()
        renderer.close()


if __name__ == "__main__":
    main()
