# Presentation: October 28, 2026

**Title: Comparing PPO, SAC, and TD3 for autonomous driving in a local simulator**

Subtitle: CARLA-oriented project with a standalone Pygame prototype.

## Five-minute talk

**0:00 Problem:** We study continuous steering, acceleration, and braking through
reinforcement learning. The objective is safe route completion, lane keeping, and
useful speed. A lightweight local simulator enables experiments without a CARLA server.

**0:40 Architecture:** Numeric observations enter a neural policy; three continuous
controls enter the environment; dynamics produce the next state and reward. Pygame
visualizes the same environment used for training. Stable-Baselines3 supplies the
algorithm implementations; our project implements the custom driving environment
and experiment pipeline.

**1:20 Reward:** Forward progress and target-speed tracking earn reward. Lane/heading
errors, jerky controls, idling, collision, and departure reduce it. Safe completion
adds a bonus. Every algorithm uses the same reward, shared with the CARLA project.

**2:00 Algorithms:**

| Algorithm | Core idea | Data | Distinctive mechanism |
|---|---|---|---|
| PPO | Conservative stochastic policy improvement | Fresh rollouts | Clipped probability-ratio objective |
| SAC | Maximize return and entropy | Replay buffer | Stochastic actor, twin critics, entropy tuning |
| TD3 | Deterministic actor with robust value estimates | Replay buffer | Twin critics, delayed actor updates, target smoothing |

PPO maximizes the minimum of unclipped and clipped probability-ratio times advantage.
This limits the incentive for large policy changes. SAC includes an entropy bonus
to encourage exploration. TD3 uses the minimum of two target critics, adds smoothing
noise to the target action, and updates the actor less often than the critics.

**3:00 Demo:** Run `python scripts/demo_local.py --policy ppo --scenario curve`.
The demo opens in 3D. Press C for chase, hood, and aerial cameras, or V for the 2D
dashboard. Explain that the 3D view displays the same flat-road dynamics and policy
state; scenery is decorative and does not add hidden obstacles.
Point out speed, steering, lane error, progress, and reward. Press 2 and 3 to compare
policies from the same seeded initial state. Press T for traffic. Press B for the
labelled reference controller. Press M and use arrows to demonstrate real control.

**4:00 Results:** Open `results/local/comparison.png` and `learning_curves.png`.
Read the actual CSV values. Explain held-out seeds and deterministic inference.
Discuss completion, collision, and lane error alongside reward. Do not infer a strong
algorithm ranking from a short, single-seed experiment.

**4:40 Limitations:** Simplified bicycle dynamics, numeric observations, simple
collision geometry, limited budget, one training seed. CARLA requires richer sensors,
complex traffic, retraining, and separate validation.

## Measured bundled run

Training seed 42, mixed scenarios, 30,000 requested transitions per algorithm
(PPO: 30,720 actual), and 12 held-out evaluation seeds 1001-1012:

| Policy | Mean return | Goal success | Collision rate |
|---|---:|---:|---:|
| PPO | 110.5 | 41.7% | 41.7% |
| SAC | 271.0 | 100% | 0% |
| TD3 | 135.8 | 58.3% | 41.7% |
| Hand-coded reference | 302.1 | 100% | 0% |

These are local 2D measurements, not CARLA results. The short budget and one training
seed limit conclusions. Training returns include stochastic exploration; deterministic
evaluation can perform better, especially for PPO. Retraining replaces these outputs,
so use the current CSV values if you run another experiment.

## Likely questions

**Is this CARLA?** This demo is a standalone Pygame simulator. The repository also
contains a separate CARLA backend that needs a running CARLA server.

**Are the algorithms from scratch?** No. They use Stable-Baselines3. Our work is the
custom simulator, task design, configuration, visualization, and benchmark pipeline.

**Why three actions?** Steering, throttle, and brake mirror CARLA controls. Braking
disables throttle through a safety interlock.

**Why ten observations?** Preview helps anticipate curves; traffic features help
following. Original CARLA observations have five features, requiring adaptation and
retraining before transfer.

**Why might the baseline outperform RL?** It encodes lane-following and distance
rules directly. RL has to discover useful behavior from reward and needs substantial
training. A short run verifies execution, not convergence.

**Is the comparison fair?** Shared task, reward, network widths, requested steps,
training seed, and evaluation seeds. PPO completes whole rollouts, so exact steps
can differ slightly. Update counts and runtime differ. More seeds and budgets are
needed for strong conclusions.

**Can it avoid obstacles?** The task supports braking and following slower same-lane
vehicles. It does not claim general overtaking or pedestrian avoidance.

## Before presenting

1. Keep the tested Python environment, `models/local`, and `results/local`.
2. Rehearse keys 1, 2, 3, B, M, T, R, and Space.
3. Keep saved charts and a screenshot available as backups.
4. State the actual simulator, interaction budget, and seeds behind the results.
