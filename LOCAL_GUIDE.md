# Local driving project

A standalone Pygame simulator now accompanies the original CARLA backend. It runs
locally without CARLA, Unreal Engine, a GPU, or a server.

## New 3D view

The demo now starts with a perspective 3D view rendered directly in Pygame. It has
3D car meshes, curved roads, trees, buildings, street lights, distance fog, a route
minimap, and telemetry. No new dependencies or graphics driver setup are required.

```powershell
python scripts/demo_local.py --view 3d --policy sac --scenario traffic
# Original dashboard view
python scripts/demo_local.py --view 2d --policy sac --scenario traffic
```

Double-click **Run_3D_Demo.bat** or **Run_Demo.bat** to open the 3D traffic demo.
Press **C** to cycle chase, hood, and aerial cameras. Press **V** to switch between
2D and 3D. Physics decisions stay at 10 Hz; display animation interpolates between
states and renders independently, targeting 60 FPS as your CPU permits.

This is a 3D visualization of the existing flat-road bicycle model. Buildings,
trees, and lamps are decorative. Vehicle collisions, road limits, rewards, and
policy inputs come from the existing environment. Your saved models and benchmark
results remain applicable; it is not CARLA or a new full 3D physics engine.

## Two-lane oncoming-traffic challenge

```powershell
python scripts/demo_local.py --view 3d --policy sac --scenario two_lane
```

Or double-click **Run_Two_Lane_Demo.bat**. Press **T** to cycle into this scenario
from any other view. It uses left-hand traffic: stay in the left 3.6 m lane, follow
slower vehicles, and avoid red cars approaching in the right lane at 25-40 km/h.
Sharper bends, lane arrows, a solid double-yellow divider, oncoming alerts, and
wrong-lane warnings make the challenge visible in both 2D and 3D.

Cars in the opposite lane move toward you and face the correct direction. Collisions
use oriented vehicle footprints in this scenario. The nearest-vehicle observation
filters for vehicles threatening the current lane, so adjacent oncoming traffic
does not incorrectly trigger a following stop. Traffic speed is signed: negative
means approaching against your direction of travel. The observation remains 10D.

Crossing into the opposite lane costs an additional -2 reward per decision. Crossing
the physical road edge terminates the episode; crossing the divider permits recovery
but exposes the car to actual oncoming collisions. Goal success requires returning
to the left lane. This is lane keeping and following, not an overtaking task.

Existing saved models can run here as a new challenge, but were trained on the old
scenarios; the demo labels that explicitly. The existing `mixed` distribution remains
straight/curve/traffic, preserving the old benchmark. To test or train this challenge:

```powershell
# Evaluate existing models on the new scenario without overwriting old results
python scripts/train_local.py --scenario two_lane --evaluate-only --episodes 12

# Train new models specifically on the challenge
python scripts/train_local.py --scenario two_lane --steps 50000 --episodes 12
```

Challenge results go to `results/local/two_lane/`; newly trained weights go to
`models/local/two_lane/`. The demo automatically prefers those weights in this
scenario when available. If no challenge weights exist, it loads the original models.
Evaluation protocol JSON records the actual source model paths. The challenge's
reward and road limits differ, so compare algorithms within this scenario rather
than comparing raw returns against the old benchmark.

## Start

From `D:\carla`:

```powershell
python -m pip install -r requirements-local.txt
python scripts/demo_local.py
```

Or double-click **Run_Demo.bat**, which starts SAC in the 3D traffic scenario.
The plain Python command defaults to the explicitly labelled hand-coded reference
controller, which also works before training.

```powershell
python scripts/demo_local.py --policy ppo --scenario curve
python scripts/demo_local.py --policy sac --scenario traffic
python scripts/demo_local.py --policy td3 --scenario straight
```

| Key | Action |
|---|---|
| 1 / 2 / 3 | PPO / SAC / TD3, resetting to the same seed |
| B | Hand-coded reference controller |
| M | Manual control with arrows or WASD |
| C | Cycle chase / hood / aerial camera in 3D |
| V | Switch 2D / 3D |
| R | Reset same episode |
| N | Next seed |
| T | Cycle straight / curve / traffic / two_lane / mixed |
| Space | Pause |
| + / - | 1x / 2x / 4x / 8x simulation speed |
| Esc | Close |

Missing models produce an explicit message. They are never replaced by a controller
under an algorithm's name. Telemetry, control bars, route progress, speed/reward traces,
and measured comparison results appear in the dashboard.

## Train and evaluate

```powershell
# Pipeline smoke test, insufficient for learning
python scripts/train_local.py --algo all --steps 2048 --episodes 3

# First experiment (also available via Train_All.bat)
python scripts/train_local.py --algo all --steps 50000 --episodes 12

# Longer training
python scripts/train_local.py --algo all --steps 200000 --episodes 30

# Evaluate saved models
python scripts/train_local.py --algo all --evaluate-only --episodes 30
```

Training uses genuine Stable-Baselines3 PPO, SAC, and TD3 implementations. This
project implements the simulator, observation design, reward integration, algorithm
configuration, training/evaluation pipeline, and presentation. Optimizers are not
implemented from scratch.

All algorithms use the same scenarios, reward, network widths, learning rate,
training seed, and requested interaction budget. PPO can exceed the requested budget
to finish its rollout; metadata records actual steps. Update counts and runtime differ.
Held-out evaluation seeds start at 1001. This is a preliminary single training-seed
experiment, not a conclusive ranking. Compare safety and completion alongside return.

Rerunning training replaces the selected algorithm's model and training log. Copy
outputs first to preserve an experiment. Run all algorithms with identical settings
for a consistent comparison.

| Output | Contents |
|---|---|
| `models/local/{ppo,sac,td3}.zip` | Learned policies |
| `models/local/<algorithm>.json` | Exact steps, seed, configuration, runtime |
| `results/local/<algorithm>/train.monitor.csv` | Actual training episodes |
| `results/local/<algorithm>_episodes.csv` | Per-episode test measurements |
| `results/local/combined_summary.csv` | Algorithm and reference comparison |
| `results/local/comparison.png` | Held-out evaluation chart |
| `results/local/learning_curves.png` | Measured training return curves |
| `results/local/evaluation_protocol.json` | Evaluation scenario and seeds |
| `results/local/REPORT.md` | Measured results with linked charts |

Generated models/results are gitignored by the original repository. Include these
folders when transferring the presentation to another computer.

## Local environment design

The kinematic bicycle model has a 2.7 m wheelbase, steering up to 0.45 radians,
throttle acceleration, brake deceleration, and drag. Each agent decision advances
0.1 simulated seconds in five substeps. Playback speed does not alter physics dt.
Roads are straight or sinusoidal. Seeds determine initial offset/heading, curve
amplitude/phase, and traffic speed. Traffic follows the same lane at lower speed;
the task includes safe following, not overtaking. Collisions use a simple axis-aligned
footprint approximation, not full rigid-body physics.

Actions: steering, throttle, brake in `[-1, 1]`. Negative throttle/brake means zero;
brake above 0.05 disables throttle. The shared `CarlaRewardCalculator` rewards
progress, target-speed tracking, centering, alignment, and smoothness. It penalizes
idling, collision, and departure; safe goal completion earns +100.

Ten normalized observations, clipped to `[-1, 1]`:

1. Speed / 50 km/h
2. Signed lateral error / 2.5 m
3. Heading error / pi
4. Remaining distance / 200 m
5. Forward bumper clearance / 50 m
6. Road heading 8 m ahead relative to vehicle / pi
7. Road heading 20 m ahead relative to vehicle / pi
8. Nearest forward traffic's relative lateral position / 5 m
9. Nearest forward traffic speed / 14 m/s
10. Previous applied steering

No forward traffic: clearance and traffic speed are 1; relative lateral position is 0.
Preview features help a feed-forward policy anticipate bends. The original CARLA
wrapper exposes five observations, so local weights cannot be loaded directly into it.

Episodes end on collision, deviation over 2.5 m, or completing the 150 m route.
Timeout is 500 decisions (50 simulated seconds). Collision/departure prevents the
goal bonus. Slower traffic can make safe following more useful than target speed.

## Repository assessment and additions

The original code contained a CARLA Gymnasium wrapper, central configuration,
route management, reward calculation, evaluation, three notebooks, and offline tests.
Its offline mode supported API testing rather than visual driving. There were no
trained weights or measured experiment results.

| Added file | Purpose |
|---|---|
| `common/local_env.py` | Dynamics, scenarios, observations, reference controller |
| `common/local_renderer.py` | Pygame dashboard and headless RGB renderer |
| `common/local_renderer3d.py` | Perspective 3D meshes, cameras, scenery and HUD |
| `scripts/train_local.py` | Shared PPO/SAC/TD3 experiment runner |
| `scripts/demo_local.py` | Interactive visual demonstration |
| `tests/test_local_env.py` | Determinism, physics, safety, rendering checks |
| `requirements-local.txt` | Dependencies without CARLA |
| `PRESENTATION_NOTES.md` | Talk track, demo sequence, questions |

The original CARLA backend remains available. Moving to CARLA needs its server,
sensor/observation adaptation, retraining, and independent evaluation. Local results
do not validate CARLA performance or real-world driving safety.

## Checks

```powershell
python -m pytest -q
python scripts/demo_local.py --policy ppo --frames 30 --screenshot results/local/demo.png
```

For headless rendering only, set `$env:SDL_VIDEODRIVER='dummy'` first. Remove this
setting for your visible presentation. Low-budget learned policies can fail or stall;
show their actual outcomes and use B for a stable reference demonstration.

Reference: [Stable-Baselines3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
