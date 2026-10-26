# CARLA Reinforcement Learning Comparison — SAC vs TD3 vs PPO

A fair, reproducible benchmark comparing three off-policy / on-policy deep RL algorithms
for autonomous driving in the [CARLA](https://carla.org/) simulator:

| Algorithm | Owner   | Algorithm family |
|-----------|---------|------------------|
| **SAC**   | Abhi    | Off-policy, maximum-entropy |
| **TD3**   | Nemo    | Off-policy, deterministic |
| **PPO**   | Rasika  | On-policy            |

All three agents train on the **same** Gymnasium environment, reward function,
observation/action spaces, route generator, simulator configuration, and evaluation
protocol, so their results can be compared directly.

> **Honesty note:** No experimental results are included in this repository. CSVs are
> generated only by actually running training + evaluation against a real CARLA server.
> Results are populated by running the notebooks/scripts, never invented.

---

## Repository layout

```
carla/
├── common/                  # Shared, environment-agnostic core (used by all three algos)
│   ├── config.py            # Centralized configuration (simulator, obs, action, reward, route, eval)
│   ├── carla_env.py         # Gymnasium CarlaDrivingEnv (reset/step/close, check_env compliant)
│   ├── reward.py            # Shared, finite reward function (CarlaRewardCalculator)
│   ├── route_manager.py     # Route generation (live CARLA map + offline mock) and progress tracking
│   └── evaluation.py        # Standardized evaluation protocol + CSV export + aggregation
├── notebooks/               # One training notebook per algorithm (identical structure)
│   ├── sac_training.ipynb
│   ├── td3_training.ipynb
│   └── ppo_training.ipynb
├── scripts/                 # CLI entry points
│   ├── check_carla_connection.py
│   ├── test_environment.py
│   └── evaluate_agent.py
├── tests/                   # pytest suite (offline unit tests + a live-server integration test)
│   ├── test_config.py
│   ├── test_reward.py
│   └── test_environment_api.py
├── models/                  # Saved models (gitignored except .gitkeep)
├── results/                 # Evaluation CSVs (gitignored except .gitkeep)
├── requirements.txt         # Local (Windows) dependencies
├── requirements-colab.txt   # Google Colab dependencies
├── pytest.ini
├── conftest.py              # Adds workspace root to sys.path; registers the carla_server marker
└── README.md
```

---

## Architecture

```
                +---------------------------------+
                |          common/config.py       |  <- single source of truth
                |  EnvConfig (sim/obs/action/route |
                |              /reward/eval)       |
                +---------------------------------+
                          |            |            |
              +-----------+            |            +-----------+
              v                        v                        v
     route_manager.py           reward.py               evaluation.py
     (route + progress)          (finite reward)         (metrics + CSV)
              |                        |                        |
              +-----------+------------+------------------------+
                          v
                   common/carla_env.py   (Gymnasium Env)
                          |
        +-----------------+-----------------+----------------+
        v                 v                 v                v
   SAC notebook      TD3 notebook     PPO notebook   evaluate_agent.py
   (stable-baselines3)
```

The three algorithms are isolated in their notebooks; everything they share lives in `common/`,
which is the contract that keeps the benchmark fair.

---

## Setup

### Prerequisites
- Python 3.10+ (developed on 3.12.6)
- A running CARLA simulator (tested against the 0.9.x line, e.g. 0.9.13–0.9.15)
- The `carla` Python package **matching your server version**

### Local (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Install the CARLA client matching your server, e.g.:
pip install carla==0.9.15
```

### Google Colab
```bash
pip install -r requirements-colab.txt
pip install carla==0.9.15   # match your server version
```
The notebooks detect Colab automatically and clone this repo + install deps on first run.

### Running CARLA
Start the simulator before training/eval. On a GPU machine:
```bash
./CarlaUE4.sh -quality-level=Low -benchmark -fps=20
```
For Colab, run CARLA on a separate GPU host (or a Colab background shell) and point
`HOST`/`PORT` in the notebook at it.

---

## Environment design

`common/carla_env.py` (`CarlaDrivingEnv`) is a standard `gymnasium.Env` validated with
`gymnasium.check_env`. It connects to a live server but also supports an offline mock mode
(`connect_on_init=False`) for unit testing the API without a simulator.

### Observation space — `Box(shape=(5,), float32)`
| Idx | Feature                        | Range        | Normalization |
|-----|--------------------------------|--------------|---------------|
| 0   | Vehicle speed                  | `[0, 1]`     | `speed_kmh / max_speed_kmh` |
| 1   | Lateral lane deviation (signed)| `[-1, 1]`    | `dev / max_lateral_dev` (- = left, + = right) |
| 2   | Heading error (signed)         | `[-1, 1]`    | `heading_err_rad / pi` |
| 3   | Distance to route goal        | `[0, 1]`     | `dist / max_goal_dist` |
| 4   | Distance to nearest vehicle   | `[0, 1]`     | `dist / max_obstacle_dist` (1.0 = clear) |

### Action space — `Box(low=-1, high=1, shape=(3,), float32)`
| Idx | Feature | Mapping |
|-----|---------|---------|
| 0 | Steering | direct to `VehicleControl.steer` |
| 1 | Throttle command | `max(0, a)` → throttle in `[0,1]` |
| 2 | Brake command | `max(0, a)` → brake in `[0,1]` |

**Safety interlock:** if `brake > 0.05`, throttle is forced to 0 (no simultaneous gas + brake).

### Reward (`common/reward.py`)
A single deterministic, finite function (`CarlaRewardCalculator`) shared by all agents:
```
R_step = w_progress*progress + w_speed*speed_factor
       + w_lateral*|dev|^1.5 + w_heading*|heading_err|
       + w_smoothness*jerk + stationary_penalty
R_terminal = collision_penalty + out_of_lane_penalty + goal_bonus
```
Weights are named parameters in `RewardConfig` (see `common/config.py`). The reward is
guaranteed finite (NaN/Inf guarded) and rewards net forward progress, target-speed tracking,
lane centering, and smooth control, while penalizing collisions, lane departures, and idling.

### Routes (`common/route_manager.py`)
- **Live mode:** builds a forward route from the spawn transform using CARLA's waypoint graph.
- **Offline/mock mode:** generates a deterministic reference trajectory for unit tests.

### Termination & time limits
- `terminated` on collision, reaching the goal, or leaving the drivable lane.
- `truncated` at `max_episode_steps` (default 500 steps @ 20 Hz = 25 s).
- `close()` destroys spawned actors and restores the original simulator settings.

---

## Training

Each `notebooks/<algo>_training.ipynb` has the same 13-section structure:

1. Dependencies & workspace setup (Colab-aware)
2. Configuration (shared `EnvConfig`)
3. CARLA connectivity check
4. Build & validate environment (`check_env`)
5. Create the model (explicit hyperparameters)
6. Checkpointing callback
7. **Smoke-test training** (1 000 steps, runs automatically)
8. **Full training** (guarded by `RUN_FULL_TRAINING = False` — does **not** auto-run)
9. Save model
10. Standardized evaluation
11. (Colab) Save artifacts to Google Drive (no hardcoded personal paths)
12. Load & demonstrate
13. TensorBoard

TensorBoard logs, periodic checkpoints, model saving, CSV export, and model loading are
included in every notebook.

> The smoke-test cell runs automatically; the long training cell only runs when you set
> `RUN_FULL_TRAINING = True`, so opening a notebook never kicks off a long experiment.

---

## Evaluation protocol

`common/evaluation.py::evaluate_policy` is the **only** evaluation path used by all three
notebooks and `scripts/evaluate_agent.py`, guaranteeing identical scenarios and metrics.

For each algorithm it runs `n_eval_episodes` (default 10) on fixed seeds and records:

| Metric | CSV column |
|--------|------------|
| Mean episode reward (± std) | `mean_reward`, `std_reward` |
| Route completion (%) | `route_completion_pct_mean` / `_std` |
| Goal success rate (%) | `goal_success_rate_pct` |
| Collision rate (%) | `collision_rate_pct` |
| Mean collisions per episode | `mean_collisions_per_episode` |
| Average speed (km/h) | `mean_speed_kmh` |
| Average lateral deviation (m) | `mean_lateral_deviation_m` |
| Episode duration (steps / s) | `mean_duration_steps`, `mean_duration_seconds` |
| Evaluation wall-clock (s) | `evaluation_wall_clock_sec` |
| Training wall-clock (s, if available) | `training_wall_clock_sec` |

**CSV schema (consistent across algorithms):**
- `<algo>_episodes.csv` — one row per episode.
- `<algo>_summary.csv` — one row with the aggregate metrics above.
- `combined_summary.csv` — produced by `aggregate_results()` for direct comparison.

---

## Scripts

```bash
# 1) Check CARLA reachability + version compatibility (clear error if missing)
python scripts/check_carla_connection.py --host localhost --port 2000

# 2) Validate the environment against a live server (runs check_env + a short rollout)
python scripts/test_environment.py --host localhost --port 2000 --steps 20

# 3) Evaluate a trained model and write CSVs
python scripts/evaluate_agent.py --model models/sac_carla.zip --algo SAC --episodes 10
```
Each script exits with a non-zero code and a clear message if CARLA is not installed, the
server is unreachable, or the client/server versions are incompatible.

---

## Testing

```bash
python -m pytest tests/ -v
```
- **Offline unit tests** (no CARLA required): configuration, reward finiteness/behavior,
  env spaces, reset/step contracts, action mapping, termination logic, determinism, and
  `gymnasium.check_env` on the offline env. These all pass locally.
- **Live integration test** (`@pytest.mark.carla_server`): real-server reset/step smoke test.
  Automatically skipped when no simulator / `carla` package is available.

The env supports `connect_on_init=False` precisely so the Gymnasium API can be validated
without a server. This validates the API contract, **not** CARLA integration — the marked
live test is the only thing that does, and it is explicitly skipped locally.

---

## Limitations & known caveats
- Observations are a low-dimensional numerical vector (no cameras/LiDAR), so policies learn
  from summary telemetry rather than raw perception.
- Reward weights are baseline starting points and should be tuned systematically; they are
  centralized in `RewardConfig` so changes apply uniformly to all algorithms.
- The environment is designed for a single ego vehicle; traffic obstacles are detected by
  proximity to other vehicles in the world.
- A real CARLA server is required for any actual training/evaluation; the offline mode is for
  API and unit testing only and does not simulate vehicle dynamics.
- Results depend on CARLA version, map, and traffic; record the exact `EnvConfig`
  (auto-saved to `<algo>_config.json`) alongside any reported metrics.

---

## License
See [LICENSE](LICENSE).
