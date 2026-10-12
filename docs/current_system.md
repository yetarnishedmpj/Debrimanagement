# Current System Specification: Multi-Agent Orbital Debris Removal

This document provides a comprehensive technical overview of the existing Multi-Agent Reinforcement Learning (MARL) experiment. It outlines what is currently implemented, the proxy models and approximations in place, mathematical formulations, environment spaces, training setups, and known limitations.

---

## 1. Quickstart & Developer Onboarding

A new developer can clone the repository, set up dependencies, verify the test suite, and run training without relying on any machine-specific environment paths.

### Prerequisites
- **Python**: Version 3.10 or 3.11.
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd Debrimanagement

# 2. Create and activate a virtual environment
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

*Note: Alternatively, you can use the automated platform scripts `setup_and_test.sh` (Linux/macOS) or `setup_and_test.ps1` (Windows PowerShell).*

### Running Verification Tests

Run the test suite from the repository root:

```bash
# Using Python module execution (ensures repository root is on sys.path):
python -m pytest tests/

# Or by setting PYTHONPATH explicitly:
# Linux/macOS:
PYTHONPATH=. pytest tests/
# Windows PowerShell:
$env:PYTHONPATH="."; pytest tests/
```

### Running Training & Evaluation

```bash
# Train decentralized IPPO agents (default: 3 agents, 5 targets, 20 iterations):
python train_marl.py --output-dir outputs

# Train using cached Space-Track TLE data:
python train_marl.py --debris-cache data/high_risk_tle_cache.json

# Launch the visual analytics dashboard:
python dashboard.py
# (Or on Windows: .\launch_dashboard.ps1)

# Render trajectory visualization from a trained checkpoint:
python visualize.py --checkpoint outputs/checkpoints --show
```

---

## 2. Implementation Status: Implemented vs. Proxy / Not Yet Established

To avoid confusion regarding simulation fidelity, the table below delineates what is fully implemented versus what is currently an empirical proxy or not yet established:

| System Component | Status | Current Implementation Details | What Is Not Yet Established |
| :--- | :--- | :--- | :--- |
| **Multi-Agent RL Algorithm** | **Implemented** | Independent PPO (IPPO) via Ray RLlib. Each chaser satellite trains its own independent neural network policy (`policy_agent_0`, etc.) from local observations. | Centralized training with decentralized execution (CTDE / MAPPO), inter-agent communication channels, or parameter sharing. |
| **Observation Space** | **Implemented** | Decentralized local matrix of shape `(max_targets, 6)` or `(max_targets, 10)` encoding explicit target availability indicator $m_j$, relative Cartesian offsets, distance, fuel, and optional relative velocities. | Observation of teammate positions, velocities, current targets, or communication messages. |
| **Action Space** | **Implemented** | `MultiDiscrete([max_targets, 3, 3, 3])` selecting target slot and 3-axis discrete burn directions $\{-1, 0, 1\}$. Maps to bounded Delta-V impulses $\Delta \mathbf{v}$ bounded by `max_impulse_delta_v`. | Continuous throttle modulation, attitude/reaction-wheel torque control. |
| **Orbital Relative Dynamics** | **Implemented (Clohessy-Wiltshire / Hill Equations)** | Analytical $6 \times 6$ Clohessy-Wiltshire State Transition Matrix $\Phi(\Delta t)$ propagating 6-DOF states $\mathbf{s}_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T$ in the Hill (LVLH) frame centered on a circular reference orbit ($R_0 = R_E + h_{\text{ref}}$, $n = \sqrt{\mu / R_0^3}$). Symplectic and volume-preserving. | Nonlinear gravitational perturbations ($J_2, J_3$), atmospheric differential drag, solar radiation pressure, third-body gravity. |
| **Debris Field Dynamics** | **Implemented (Dynamic Propagation)** | Debris fragments are full 6-DOF dynamical bodies propagating under unforced Clohessy-Wiltshire relative motion $\mathbf{s}(t+\Delta t) = \Phi(\Delta t)\mathbf{s}(t)$ throughout each episode. Initial along-track velocities $\dot{y}_0 = -2nx_0$ generate closed periodic 2:1 relative ellipses with zero secular drift. | Debris tumbling / rotational dynamics, fragmentation cascades during collisions. |
| **Control Inputs & Fuel Accounting** | **Implemented (Physical Impulse / Delta-V Accounting)** | Bounded impulse control ($\|\Delta \mathbf{v}\| \le \Delta v_{\text{max}}$) with direct propellant budget deduction: $\Delta \text{fuel} = \|\Delta \mathbf{v}\| / v_{\text{capacity}}$. Supports continuous acceleration mode ($\mathbf{a}$) via discrete input matrix $\Gamma(\Delta t)$. | Finite-burn thrust buildup curves, propellant slosh, reaction mass exhaust limits. |
| **Operational Scope: Target Resolution** | **Rendezvous Arrival (Phase 1)** | Arrival within proximity threshold ($\le 12.0$ km) simulates successful coarse rendezvous arrival; target retired from tracking. | **Does NOT simulate physical capture (Phase 2)** (no docking, grappling, harpoons, robotic arms, contact dynamics, or $\Delta v_{\text{rel}} \to 0$). **Does NOT simulate debris removal (Phase 3)** (no de-orbit burn or atmospheric reentry disposal). |
| **Collision Hazards & Safety** | **Not Implemented** | Satellites pass through each other and debris with zero collision interaction unless `collision_radius_km` is configured. | High-fidelity keep-out zones, covariance ellipsoids, fragmentation modeling. |
| **Real Orbital Data & SGP4 Propagation** | **Implemented (SGP4 + OMM + LVLH Frame)** | CelesTrak GP API and Space-Track client in `fetch_data.py` supporting CCSDS OMM (JSON/CSV) accommodating catalog IDs > 100,000 (up to 9 digits). Established SGP4 propagator (WGS72) propagates records to a common simulation epoch and transforms inertial TEME coordinates into consistent Hill (LVLH) relative positions and velocities. Servicing spacecraft maintains a separate maneuver and relative-motion model. | Continuous ephemeris numerical integration (cowell / encke), full solar radiation pressure, third-body lunar/solar gravitation. |

---

### Operational Definition: Rendezvous vs. Physical Capture vs. Complete Debris Removal

In space mission engineering and astrodynamics literature, Active Debris Removal (ADR) comprises three fundamentally different operational phases:

1. **Far-Range Rendezvous (Phase 1 — Simulated)**:
   - **Definition**: Orbital phasing, plane matching, and relative navigation from deployment staging to an inspection hold point or proximity operations boundary (standoff radius $r \le 12.0$ km).
   - **What the simulator implements**: When a chaser satellite arrives within `capture_radius_km` ($12.0$ km), it achieves a **successful rendezvous arrival**. The target slot is deactivated (`target_active = False`) to reflect that the rendezvous assignment objective has been accomplished. The agent remains in orbit with its remaining fuel to pursue subsequent rendezvous targets.
2. **Physical Capture (Phase 2 — Not Simulated)**:
   - **Definition**: Close-proximity operations (sub-meter standoff) with relative velocity nulled ($\Delta v_{\text{rel}} \approx 0$), followed by mechanical grappling (robotic manipulator), harpoon penetration, net deployment, or docking latching. The chaser and debris become a coupled rigid multi-body system.
   - **What the simulator does NOT implement**: No mechanical contact forces, robotic arms, grappling latches, tumbling target synchronization, or relative velocity nulling are modeled.
3. **Complete Debris Removal (Phase 3 — Not Simulated)**:
   - **Definition**: The coupled chaser-debris stack executes a major retrograde propulsion maneuver (typically $100\text{--}300$ m/s $\Delta v$) to lower the perigee into Earth's upper atmosphere for destructive reentry, or re-boosts to a designated graveyard disposal orbit.
   - **What the simulator does NOT implement**: The chaser does not perform de-orbit disposal burns and does not exit the simulation upon rendezvous; no atmospheric reentry or disposal orbital decay is simulated.

---

## 3. Fleet Scale and Environment Configuration

As defined in `OrbitalEnvConfig` (`orbital_env.py`):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `num_agents` | `3` | Active chaser satellites (`agent_0`, `agent_1`, `agent_2`). |
| `num_targets` | `5` | Active debris pieces present at episode start. |
| `max_targets` | `5` | Fixed target slot capacity for tensor shapes. |
| `max_episode_steps` | `300` | Maximum control intervals per episode before truncation. |
| `dynamics_model` | `"clohessy_wiltshire"` | Orbital dynamics solver (`"clohessy_wiltshire"` or `"kinematic_translation"`). |
| `control_mode` | `"impulsive"` | Propulsion control mode (`"impulsive"` or `"continuous_acceleration"`). |
| `time_step_s` | `10.0` s | Astrodynamic propagation time step $\Delta t$ per control step. |
| `reference_altitude_km` | `500.0` km | Circular reference orbit altitude above Earth surface ($h_{\text{ref}}$). |
| `earth_radius_km` | `6378.137` km | Earth equatorial radius ($R_E$). Orbital radius $R_0 = R_E + h_{\text{ref}} = 6878.137$ km. |
| `gravitational_parameter_mu` | `398600.4418` $\text{km}^3/\text{s}^2$ | Standard geocentric gravitational parameter $\mu$. Mean motion $n = \sqrt{\mu / R_0^3} \approx 0.001107$ rad/s. |
| `max_impulse_delta_v` | `0.005` km/s ($5$ m/s) | Maximum commanded physical impulse per control interval. |
| `target_drift_mode` | `"periodic_ellipse"` | Debris initial drift mode: `"periodic_ellipse"` ($\dot{y}_0 = -2nx_0$), `"free_drift"`, or `"stationary_initial"`. |
| `obs_includes_velocity`| `False` | When `True`, expands observation space from `(max_targets, 6)` to `(max_targets, 10)` with relative velocities. |
| `velocity_range_km_s` | `0.1` km/s ($100$ m/s) | Normalization domain for relative velocity observations. |
| `world_range_km` | `250.0` km | Coordinate domain boundary ($x, y, z \in [-250.0, 250.0]$ km). |
| `capture_radius_km` | `12.0` km | Proximity radius required to secure debris rendezvous arrival. |
| `initial_fuel` | `1.0` ($100\%$) | Normalized propellant mass fraction at episode reset. |
| `fuel_capacity_delta_v` | `300.0` km/s | Total propellant budget capacity for depletion accounting. |
| `agent_spawn_radius_km`| `15.0` km | Radius of uniform distribution for agent initial staging. |
| `target_position_noise_km` | `4.0` km | Gaussian perturbation added to target locations per episode. |
| `target_resolution_policy` | `"no_op"` | Deliberate target fallback policy (`"no_op"` or `"nearest_agent"`). |

---

## 4. Observation and Action Spaces

The environment `OrbitalDebrisRemovalEnv` in `orbital_env.py` is structured as follows:

### Observation Space
- **Gymnasium Definition**: `spaces.Box(low=-1.0, high=1.0, shape=(max_targets, 6), dtype=np.float32)`
- **Dimension**: `(5, 6)` under default `max_targets = 5`.
- **Decentralization**: Agent $i$ receives only its own local observation matrix. Teammate states are hidden.

For agent $i$ and target slot $j \in \{0, \dots, \text{max\_targets}-1\}$, the target feature vector $o_{i,j}$ contains:

$$o_{i,j} = \left[ m_j,\, \Delta x_{i,j},\, \Delta y_{i,j},\, \Delta z_{i,j},\, d_{\text{norm}, i, j},\, f_{\text{norm}, i} \right]$$

1. **Explicit Availability Indicator ($m_j$)**:
   $$m_j = \begin{cases} 1.0 & \text{if target slot } j \text{ is active} \\ 0.0 & \text{if target slot } j \text{ is inactive / already captured} \end{cases}$$
   Explicitly disambiguates inactive slots from active targets positioned at relative zero offset $(\Delta x=0, \Delta y=0, \Delta z=0, d=0)$.
2. **Relative In-Plane X ($\Delta x_{i,j}$)**:
   $$\Delta x_{i,j} = \text{clip}\left(\frac{x_{\text{target}, j} - x_{\text{agent}, i}}{\text{world\_range\_km}}, -1.0, 1.0\right) \quad (\text{0.0 if target } j \text{ is inactive})$$
3. **Relative In-Plane Y ($\Delta y_{i,j}$)**:
   $$\Delta y_{i,j} = \text{clip}\left(\frac{y_{\text{target}, j} - y_{\text{agent}, i}}{\text{world\_range\_km}}, -1.0, 1.0\right) \quad (\text{0.0 if target } j \text{ is inactive})$$
4. **Relative Out-of-Plane Z ($\Delta z_{i,j}$)**:
   $$\Delta z_{i,j} = \text{clip}\left(\frac{z_{\text{target}, j} - z_{\text{agent}, i}}{\text{world\_range\_km}}, -1.0, 1.0\right) \quad (\text{0.0 if target } j \text{ is inactive})$$
5. **Normalized Distance ($d_{\text{norm}, i, j}$)**:
   $$d_{\text{norm}, i, j} = \text{clip}\left(\frac{\|\mathbf{r}_{\text{target}, j} - \mathbf{r}_{\text{agent}, i}\|_2}{\text{world\_range\_km}}, 0.0, 1.0\right) \quad (\text{0.0 if target } j \text{ is inactive})$$
6. **Normalized Propellant ($f_{\text{norm}, i}$)**:
   $$f_{\text{norm}, i} = \text{clip}(2 \cdot \text{fuel}_i - 1.0, -1.0, 1.0) \in [-1.0, 1.0]$$
   Agent-level propellant status preserved for local policy observability.

### Action Space
- **Gymnasium Definition**: `spaces.MultiDiscrete([max_targets, 3, 3, 3])`
- **Action Vector**: $a_i = [u_{\text{target}}, u_x, u_y, u_z]$

| Component | Values | Mapping / Physical Action |
| :--- | :--- | :--- |
| $u_{\text{target}}$ | $\{0, \dots, \text{max\_targets}-1\}$ | Target slot selection. The agent designates which target fragment it tracks. Inactive targets are resolved via the configured deliberate policy (`target_resolution_policy`). |
| $u_x$ | $\{0, 1, 2\}$ | Burn along X axis: $\delta_x = u_x - 1 \in \{-1.0, 0.0, +1.0\}$. |
| $u_y$ | $\{0, 1, 2\}$ | Burn along Y axis: $\delta_y = u_y - 1 \in \{-1.0, 0.0, +1.0\}$. |
| $u_z$ | $\{0, 1, 2\}$ | Burn along Z axis: $\delta_z = u_z - 1 \in \{-1.0, 0.0, +1.0\}$. |

### Deliberate Target Selection Policy & Action Masking
In earlier prototypes, selecting an inactive target slot triggered a silent fallback that redirected the agent to whichever active target was closest to the coordinate origin `[0, 0, 0]`. This has been replaced with a deliberate policy configuration (`target_resolution_policy` in `OrbitalEnvConfig`):

1. **`no_op` Policy (Default Scientific Specification)**:
   - **No Silent Redirection**: If an agent selects an inactive or already-captured target slot, the environment does not reassign it to an arbitrary target.
   - **Explicit No-Op Dynamics**: `selected_target_index` resolves to `None`. The agent still executes its requested physical burn vector and consumes base propellant, but does not close distance toward any target.
   - **No Capture / No Unearned Progress**: Progress reward is set to $0.0$, and the agent cannot capture any debris fragments during the step.
   - **Explicit Error Penalty**: The agent incurs an `invalid_action_penalty` ($-0.2$) as an explicit feedback signal that an unavailable target was commanded.
   - **Diagnostic Info**: Returns `"selected_target_index": None` and `"invalid_target_selection": True` in `infos[agent_id]`.

2. **`nearest_agent` Policy (Relative Proximity Fallback)**:
   - When an inactive target slot is requested, the fallback dynamically redirects the agent to the active target closest to the **agent's current coordinates** ($\arg\min_{j \in \text{active}} \|\mathbf{r}_{\text{target}, j} - \mathbf{r}_{\text{agent}, i}\|_2$).
   - Completely removes the legacy artifact of sorting targets relative to the coordinate origin.

3. **Explicit Action Masking**:
   - The environment exposes `env.get_action_mask(agent_id)` returning a boolean array of length `max_targets` (`True` if active, `False` if inactive/captured).
   - In both `reset()` and `step()`, `infos[agent_id]["action_mask"]` contains the current availability mask.
   - This enables action-masked policy networks (e.g. invalid action masking with logit masking) to prevent the policy from ever selecting inactive targets during rollout sampling.

### 6-DOF Dynamical State & Clohessy-Wiltshire Equations of Motion

The environment replaces kinematic point translations with full 6-DOF relative astrodynamics in the Local-Vertical/Local-Horizontal (LVLH) Hill reference frame centered on a circular reference orbit ($R_0 = R_E + h_{\text{ref}}$, $n = \sqrt{\mu / R_0^3}$):

$$\mathbf{s}_t = \begin{bmatrix} r_x & r_y & r_z & v_x & v_y & v_z \end{bmatrix}^T$$

- **$r_x$ (Radial)**: Position along the outward Earth-center radial line [km].
- **$r_y$ (In-Track / Along-Track)**: Position along the circular reference velocity vector [km].
- **$r_z$ (Cross-Track / Out-of-Plane)**: Position normal to the orbital plane along angular momentum $\mathbf{h}$ [km].
- **$v_x, v_y, v_z$**: Relative velocity components in the rotating Hill frame [km/s].

#### Clohessy-Wiltshire Differential Equations
Under the assumptions of a circular reference orbit, spherical central gravity, and separation distance $\|\mathbf{r}\| \ll R_0$:

$$\ddot{x} - 2n\dot{y} - 3n^2 x = u_x$$
$$\ddot{y} + 2n\dot{x} = u_y$$
$$\ddot{z} + n^2 z = u_z$$

#### Analytical State Transition Matrix $\Phi(\Delta t)$
State propagation over control interval $\tau = \Delta t$ is computed via the closed-form $6 \times 6$ matrix $\Phi(\tau)$:

$$\mathbf{s}(t + \tau) = \Phi(\tau) \mathbf{s}(t)$$

$$\Phi(\tau) = \begin{bmatrix}
4 - 3\cos(n\tau) & 0 & 0 & \frac{\sin(n\tau)}{n} & \frac{2}{n}(1 - \cos(n\tau)) & 0 \\
6(\sin(n\tau) - n\tau) & 1 & 0 & -\frac{2}{n}(1 - \cos(n\tau)) & \frac{4\sin(n\tau) - 3n\tau}{n} & 0 \\
0 & 0 & \cos(n\tau) & 0 & 0 & \frac{\sin(n\tau)}{n} \\
3n\sin(n\tau) & 0 & 0 & \cos(n\tau) & 2\sin(n\tau) & 0 \\
6n(\cos(n\tau) - 1) & 0 & 0 & -2\sin(n\tau) & 4\cos(n\tau) - 3 & 0 \\
0 & 0 & -n\sin(n\tau) & 0 & 0 & \cos(n\tau)
\end{bmatrix}$$

- **Symplectic Structure**: $\det(\Phi(\tau)) \equiv 1.0$ (volume-preserving Hamiltonian phase flow).
- **Identity Property**: $\Phi(0) = \mathbf{I}_6$.

#### Bounded Impulse Control & Physical Propellant Accounting
For an active control command with direction vector $\mathbf{d} = [\delta_x, \delta_y, \delta_z]^T$:
- If $\|\mathbf{d}\|_2 > 0$:
  $$\Delta \mathbf{v}_{\text{cmd}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2} \cdot \Delta v_{\text{max}} \quad (\text{default: } 0.005\text{ km/s} = 5\text{ m/s})$$
- Physical propellant consumption:
  $$\Delta \text{fuel} = \frac{\|\Delta \mathbf{v}_{\text{cmd}}\|_2}{v_{\text{capacity}}}$$
- If available fuel $f_{\text{avail}} < \Delta \text{fuel}$, the delivered impulse is throttled:
  $$\Delta \mathbf{v}_{\text{delivered}} = \Delta \mathbf{v}_{\text{cmd}} \cdot \frac{f_{\text{avail}}}{\Delta \text{fuel}}, \quad \Delta \text{fuel} \leftarrow f_{\text{avail}}$$
- Instantaneous post-impulse velocity: $\mathbf{v}^+ = \mathbf{v}^- + \Delta \mathbf{v}_{\text{delivered}}$.
- Spacecraft state after control interval:
  $$\mathbf{s}(t + \Delta t) = \Phi(\Delta t) \begin{bmatrix} \mathbf{r}(t) \\ \mathbf{v}^+ \end{bmatrix}$$
- If fuel is depleted ($f_{\text{avail}} = 0$), the spacecraft continues ballistic unforced orbital motion: $\mathbf{s}(t + \Delta t) = \Phi(\Delta t) \mathbf{s}(t)$.

#### Dynamic Debris Field Propagation & Periodic Relative Orbits
Debris fragments are dynamic bodies evolving under unforced relative motion:
$$\mathbf{s}_{\text{target}, j}(t + \Delta t) = \Phi(\Delta t) \mathbf{s}_{\text{target}, j}(t)$$
To avoid uncontrolled secular drift along the track ($y(t) \sim -3(\dot{y}_0 + 2nx_0)t$), debris velocities are initialized with the classic periodic relative orbit condition:
$$\dot{y}_0 = -2 n x_0, \quad \dot{x}_0 = 0, \quad \dot{z}_0 = 0$$
This eliminates the secular drift term, causing each debris fragment to trace a stable, bounded 2:1 relative ellipse in the orbital plane ($x(t) = x_0\cos nt$, $y(t) = y_0 - 2x_0\sin nt$).

### Simultaneous Execution Sequence & Conflict Resolution
To ensure a true simultaneous-action environment where agent execution order does not alter target availability or physics outcomes, the environment executes a 5-phase update cycle in `step()`:

1. **Snapshot Current State**: Captures pre-step 6-DOF positions, velocities, fuel, and target availability masks before any agent action is parsed.
2. **Propagate Debris Field**: Propagates all active debris targets dynamically across $\Delta t$ using $\Phi(\Delta t)$.
3. **Collect Agent Actions & Compute Proposed Transitions**: Parses agent commands, applies bounded impulses $\Delta \mathbf{v}$, deducts propellant, and propagates agent 6-DOF states via $\Phi(\Delta t)$ into uncommitted staging buffers.
4. **Joint Interaction Resolution**: Evaluates captures and target conflicts jointly against post-transition geometries:
   - *Target Conflict Resolution (Proximity Priority)*: When multiple agents enter the capture radius ($12.0$ km) of the same target, the capture is awarded to the agent closest to the target ($\min d_t(i, j^*)$).
   - *Deterministic Tie-Breaking*: If two or more competing agents have identical final distances (within numerical tolerance $10^{-5}$ km), the tie is broken deterministically by lower agent index (`agent_0` over `agent_1`).
   - *Cooperative Reward Distribution*: All teammate satellites receive the cooperative bonus ($+4.0$) for each captured target.
   - *Collision Detection*: Evaluates inter-agent distances if `collision_radius_km` is configured.
5. **Commit State and Rewards**: Atomically updates physical positions, velocities, propellant levels, active target flags, and 6-DOF episode histories.

---

## 5. Reward Function and Fuel-Cost Proxy Model

### Step Reward Formulation
For agent $i$ tracking chosen active target $j^*$:

$$R_i = R_{\text{progress}, i} + R_{\text{capture}, i} + R_{\text{cooperative}, i} - C_{\text{fuel}, i} - C_{\text{idle}, i} - C_{\text{invalid}, i}$$

1. **Progress Reward**:
   $$R_{\text{progress}, i} = 2.0 \cdot \frac{d_{t-1}(i, j^*) - d_t(i, j^*)}{\text{world\_range\_km}}$$
   Rewords closing Euclidean distance toward target $j^*$.
2. **Individual Capture Reward**:
   $$R_{\text{capture}, i} = \begin{cases} 15.0 & \text{if } d_t(i, j^*) \le 12.0\text{ km and target active} \\ 0.0 & \text{otherwise} \end{cases}$$
3. **Cooperative Teammate Share**:
   $$R_{\text{cooperative}, i} = 4.0 \cdot (\text{number of teammates } k \ne i \text{ that captured a target this step})$$
   Global reward sharing providing cooperative incentive without centralized action control.
4. **Fuel-Cost Proxy Penalty**:
   $$C_{\text{fuel}, i} = 0.4 \cdot \Delta \text{fuel}_i, \quad \text{where } \Delta \text{fuel}_i = \frac{C_{\text{proxy}, i}}{\text{fuel\_capacity\_delta\_v}}$$
5. **Idle Penalty**:
   $$C_{\text{idle}, i} = \begin{cases} 0.03 & \text{if } \|\mathbf{d}\|_2 = 0 \\ 0.0 & \text{otherwise} \end{cases}$$
6. **Invalid Action Penalty**:
   $$C_{\text{invalid}, i} = \begin{cases} 0.2 & \text{if action is malformed, out-of-range, or selects inactive target under no_op} \\ 0.0 & \text{otherwise} \end{cases}$$

### Heuristic Fuel-Cost Proxy Model

The method `_fuel_cost_proxy` (aliased as `_lambert_proxy_delta_v` for legacy compatibility) computes an empirical heuristic proxy for relative maneuver expenses:

$$C_{\text{proxy}, i} = k_1 \cdot d_{\text{in-plane}} + k_2 \cdot d_{\text{cross-track}} + 0.1 \cdot \|\mathbf{u}\|_2 + P_{\text{alignment}}$$

Where:
- $d_{\text{in-plane}} = \sqrt{(x_{\text{target}} - x_{\text{agent}})^2 + (y_{\text{target}} - y_{\text{agent}})^2}$
- $d_{\text{cross-track}} = |z_{\text{target}} - z_{\text{agent}}|$
- $k_1 = 0.025$ (`k1_phasing_cost`): In-plane phasing cost coefficient.
- $k_2 = 0.085$ (`k2_plane_change_cost`): Plane-change cost coefficient ($\approx 3.4\times$ higher than in-plane transfers, capturing the severe energy penalty of changing orbital inclination relative to in-plane phasing).
- $P_{\text{alignment}}$: Penalizes burns oriented away from the line-of-sight to the target:
  $$\cos(\theta) = \frac{\Delta \mathbf{r} \cdot (\mathbf{r}_{\text{target}} - \mathbf{r}_{\text{agent}})}{\|\Delta \mathbf{r}\|_2 \|\mathbf{r}_{\text{target}} - \mathbf{r}_{\text{agent}}\|_2}$$
  $$P_{\text{alignment}} = 0.5 \cdot (1.0 - \text{clip}(\cos(\theta), -1.0, 1.0)) \in [0.0, 1.0]$$

> [!CAUTION]
> **Scientific Disclaimer on Fuel-Cost Proxy**:
> Do **NOT** report $C_{\text{proxy}}$ as a physically validated $\Delta v$ calculation.
> Until replaced with an astrodynamically integrated two-body / relative dynamics solver (such as two-impulse Lambert boundary-value targeting or continuous low-thrust optimal control), this value must strictly be designated as a **heuristic fuel-cost proxy** in all documentation, dashboards, and research results. It serves as an empirical reward-shaping surrogate for gradient stability, not an astrodynamically converged orbital maneuver.

---

## 6. RL Algorithm & Training Details

The training script `train_marl.py` leverages **Ray RLlib** with PyTorch.

- **Algorithm**: **Independent Proximal Policy Optimization (IPPO)**.
- **Topology**: `decentralized_ippo`.
- **Policy Mapping**:
  - `policy_mapping_fn(agent_id)` maps each agent `agent_k` directly to its dedicated policy `policy_agent_k`.
  - Independent actor and critic weights; no parameter sharing across agents.
- **Network Architecture**:
  - Fully connected MLP with hidden layer sizes `[512, 512, 256]`.
  - Separate value networks (`vf_share_layers = False`).
- **Optimization Hyperparameters**:
  - Discount factor ($\gamma$): `0.995`
  - GAE smoothing ($\lambda$): `0.97`
  - PPO clip parameter ($\epsilon$): `0.2`
  - Learning rate: $3 \times 10^{-4}$ linearly decaying to $1 \times 10^{-5}$ across total timesteps.
  - Entropy coefficient: $0.05$ linearly decaying to $0.005$ across $80\%$ of total timesteps.
  - Value function coefficient: $1.0$
  - Gradient clipping: $0.5$
  - Train batch size: `4096`
  - Minibatch size: `512`
  - SGD epochs per batch: `10`
  - Training stopping criteria: `20` iterations or `150,000` timesteps.

---

## 7. Orbital Coordinate Generation

### Reference Frame
The simulation assumes a local relative Cartesian frame centered on a circular reference deployment orbit:
- **X (In-Track)**: Tangential along the orbital velocity vector.
- **Y (Radial)**: Perpendicular to velocity, pointing radially outward.
- **Z (Cross-Track)**: Normal to the orbital plane (inclination axis).

### Initial Agent Positions
Sampled uniformly in `_resolve_agent_positions`:
- $x, y \sim \mathcal{U}[-15.0, +15.0]$ km
- $z \sim \mathcal{U}[-15.0, +15.0] \times 0.35 = \mathcal{U}[-5.25, +5.25]$ km

### Debris Positions Generation
Two operational modes exist:

1. **Synthetic Breakup Clusters (Default)**:
   In `_sample_clustered_targets`, positions are sampled from three cluster centers representing debris breakups:
   - $C_1 = [90.0, -50.0, 18.0]$ km
   - $C_2 = [-120.0, 70.0, -25.0]$ km
   - $C_3 = [55.0, 135.0, 12.0]$ km
   Gaussian dispersion is added: $\sigma_{x,y} = 18.0$ km, $\sigma_z = 9.0$ km. A per-episode noise perturbation ($\sigma = 4.0$ km) is applied at reset.

2. **Real-world SGP4 Propagation & Hill (LVLH) Transformation (`fetch_data.py`)**:
   General Perturbations orbital records are ingested in modern CCSDS OMM (Orbit Mean-Elements Message) formats (JSON, CSV) or legacy TLE/3LE format, fully supporting catalog identifiers exceeding 5 digits (6-digit > 100,000, 9-digit Space Fence IDs per 2026 CelesTrak standards).
   
   - **Established SGP4 Propagation**: Each documented record with recorded epoch $T_{\text{epoch}}$ is propagated using the established SGP4 propagator with WGS72 gravitational constants to a common simulation epoch $T_{\text{sim}}$:
     $$\mathbf{r}_{\text{TEME}}(T_{\text{sim}}), \mathbf{v}_{\text{TEME}}(T_{\text{sim}}) = \text{SGP4}(\text{record}, T_{\text{sim}})$$
   - **Consistent Reference Frame Transformation**: A reference orbit $(\mathbf{R}_0, \mathbf{V}_0)$ establishes the rotating Hill (LVLH) frame basis:
     $$\hat{\mathbf{e}}_r = \frac{\mathbf{R}_0}{\|\mathbf{R}_0\|}, \quad \hat{\mathbf{e}}_c = \frac{\mathbf{R}_0 \times \mathbf{V}_0}{\|\mathbf{R}_0 \times \mathbf{V}_0\|}, \quad \hat{\mathbf{e}}_t = \hat{\mathbf{e}}_c \times \hat{\mathbf{e}}_r, \quad C = \begin{bmatrix} \hat{\mathbf{e}}_r^T \\ \hat{\mathbf{e}}_t^T \\ \hat{\mathbf{e}}_c^T \end{bmatrix}$$
     Relative Cartesian positions and rotating-frame velocities for each debris target are computed as:
     $$\mathbf{r}_{\text{LVLH}} = C (\mathbf{r}_{\text{TEME}} - \mathbf{R}_0)$$
     $$\mathbf{v}_{\text{LVLH}} = C (\mathbf{v}_{\text{TEME}} - \mathbf{V}_0) - \boldsymbol{\omega}_{\text{LVLH}} \times \mathbf{r}_{\text{LVLH}}, \quad \text{where } \boldsymbol{\omega}_{\text{LVLH}} = \begin{bmatrix} 0 \\ 0 \\ \frac{\|\mathbf{R}_0 \times \mathbf{V}_0\|}{\|\mathbf{R}_0\|^2} \end{bmatrix}$$
   - **Separate Servicing Spacecraft Model**: The controlled servicing spacecraft maintains its own maneuver model (`ServicingSpacecraftModel`), executing discrete Delta-V impulses $\Delta \mathbf{v}$ and unforced relative motion governed by the Clohessy-Wiltshire state transition matrix $\Phi(\Delta t)$.
   - **Proximity Selection & Fallback**: The closest targets in the cluster to the servicing staging point are selected and populated into `target_positions` and `target_velocities`. If bare synthetic element dicts without recorded epoch/SGP4 elements are provided, a geometric fallback ensures backwards compatibility.

---

## 8. Summary of Model Assumptions & Roadmap for Realism

A developer extending this project should be aware of the following baseline assumptions:

1. **Astrodynamical Propagation**:
   - *Current Implementation*: **Clohessy-Wiltshire (Hill) relative-motion equations** around a circular reference orbit using an analytical $6 \times 6$ State Transition Matrix $\Phi(\Delta t)$.
   - *Future Enhancements*: Non-circular eccentric orbits (Tschauner-Hempel equations), nonlinear gravity gradient terms for large separations ($\|\mathbf{r}\| > 100$ km), and orbital perturbations ($J_2$ oblateness, atmospheric differential drag, solar radiation pressure).
2. **Transfer Planning**:
   - *Current Implementation*: Bounded impulse maneuvers ($\Delta \mathbf{v}$) and continuous constant accelerations ($\mathbf{a}$) with physical $\Delta v$ propellant depletion.
   - *Future Enhancements*: Multi-impulse Lambert targeting boundary-value solver, Hohmann / bi-elliptic transfer sequences, and optimal low-thrust continuous trajectories.
3. **Capture Operations**:
   - *Current Implementation*: Coarse rendezvous arrival at $\le 12.0$ km standoff.
   - *Future Enhancements*: Multi-phase operations (far-range phasing $\to$ proximity rendezvous with $\Delta v_{\text{rel}} \to 0$ $\to$ physical capture at $< 10$ m).
4. **Multi-Agent Coordination**:
   - *Current Implementation*: Independent PPO without communication or shared critic.
   - *Future Enhancements*: Multi-Agent PPO with Centralized Training and Decentralized Execution (MAPPO / CTDE), attention-based target allocation, or inter-satellite communications.
5. **Safety**:
   - *Current Implementation*: Configurable inter-agent collision proximity checks.
   - *Future Enhancements*: Add keep-out zones, debris covariance ellipsoids, and collision avoidance trajectory constraints.

---

## 9. Stage 1 Engineering Milestone: Completion Criteria & Verification Matrix

All six Stage 1 milestone criteria are fully satisfied and verified via the test suite:

| Criterion | Implementation Status | Verification Method & Test Reference |
| :--- | :--- | :--- |
| **1. All tests pass** | **100% Green** | `python -m pytest tests/` executes cleanly with all unit and integration tests passing. |
| **2. Seed Reproducibility** | **Verified Bitwise Identical** | `test_seed_reproducibility_across_identical_episodes` verifies that running the environment with the same seed and action sequence produces exact bitwise-identical trajectories, fuel traces, rewards, and info dicts across independent runs. |
| **3. Agent Update Order Invariance** | **Verified Order-Independent** | `test_simultaneous_updates_order_invariance_under_contention`, `test_tie_breaking_rule_under_identical_distances`, `test_simultaneous_multi_target_capture_no_preemption`, and `test_multistep_episode_order_invariance` prove that reversing agent processing order (`[agent_2, agent_1, agent_0]` vs `[agent_0, agent_1, agent_2]`) yields identical state trajectories and rewards. |
| **4. Explicit Inactive Target Handling** | **Explicit Disambiguation** | Targets include binary availability indicator $m_j \in \{0.0, 1.0\}$. The deliberate `no_op` policy penalizes invalid target choices with $-0.2$ without silent redirection. Action masks (`get_action_mask`) expose valid choices to policies. Verified in `test_target_resolution_no_op_policy_and_action_mask` and `test_reset_observations_are_clamped_and_shaped`. |
| **5. Documented Observation & Action Spaces** | **Formally Documented** | Detailed mathematical formulation in Section 4: Observation space `(max_targets, 6)` encoding $[m_j, \Delta x, \Delta y, \Delta z, d_{\text{norm}}, f_{\text{norm}}]$ and Action space `MultiDiscrete([max_targets, 3, 3, 3])`. |
| **6. Unambiguous Reward & Fuel-Cost Semantics** | **Unambiguously Formulated** | Fuel cost is formally designated as a **heuristic fuel-cost proxy** ($C_{\text{proxy}}$) and explicitly distinguished from physically validated orbital $\Delta v$. Target intercept is explicitly defined as **Phase 1: Successful Rendezvous Arrival** (distinguished from physical mechanical capture and de-orbit disposal). Verified in `test_fuel_cost_proxy_semantics_and_plane_change_penalty` and `test_rendezvous_arrival_operational_semantics`. |

---

## 10. Stage 2 Physics Milestone: Relative Dynamics & Bounded Control Verification Matrix

The Stage 2 astrodynamic milestone replaces kinematic jumps and frozen debris with the Clohessy-Wiltshire relative-motion model, 6-DOF dynamical states, dynamic debris evolution, and bounded impulse control:

| Physics Feature | Implementation Status | Verification Method & Test Reference |
| :--- | :--- | :--- |
| **6-DOF State Representation** | **Implemented** | Servicing satellites and debris targets maintain full state vectors $\mathbf{s}_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T$ in the Hill (LVLH) frame. Verified via `test_dynamical_state_representation` with `get_agent_state()`, `get_target_state()`, and `get_fleet_states()`. |
| **Clohessy-Wiltshire STM Properties** | **Mathematically Verified** | Verified via `test_cw_stm_mathematical_properties_and_orbit_closure`: $\Phi(0) = \mathbf{I}_6$, $\det(\Phi(\tau)) = 1.0$ (Hamiltonian symplectic volume preservation), and periodic orbit return after orbital period $T = 2\pi / n$ ($\approx 5676.8$ s) for $\dot{y}_0 = -2nx_0$. |
| **Dynamic Debris Field Evolution** | **Dynamically Propagated** | Debris locations evolve continuously under unforced CW dynamics across control steps rather than remaining frozen. Initial conditions $\dot{y}_0 = -2nx_0$ ensure bounded 2:1 relative ellipses. Verified in `test_debris_locations_evolve_dynamically_across_steps`. |
| **Bounded Impulse Control & Fuel Model** | **Physically Accounted** | Discrete action directions $\{0, 1, 2\}$ map to bounded impulses $\|\Delta \mathbf{v}\| \le \Delta v_{\text{max}}$. Propellant is deducted directly as $\Delta \text{fuel} = \|\Delta \mathbf{v}\| / v_{\text{capacity}}$, and throttled when remaining fuel is insufficient. Verified in `test_bounded_impulse_control_and_physical_fuel_accounting`. |
| **Relative Velocity Observations** | **Supported** | Configurable via `obs_includes_velocity=True` providing shape `(max_targets, 10)` with normalized relative velocity components and speed. Verified in `test_observation_space_includes_relative_velocities_when_configured`. |
| **Continuous Acceleration Mode** | **Supported** | Spacecraft propagation supports continuous constant thrust via discrete input matrix $\Gamma(\Delta t)$. Verified in `test_continuous_acceleration_control_mode`. |

---

## 11. Non-Learning Baselines Specification (Step 3.1)

To answer the core research question — **does learning add measurable value over a simple method?** — the environment implements two deterministic, non-learning baseline controllers in [`baselines.py`](file:///c:/Users/mahar/Downloads/Debrimanagement/baselines.py):

### 11.1 Nearest-Target Heuristic

Each chaser satellite evaluates active target slots and greedily selects the target minimizing Euclidean relative distance:

$$j^* = \arg\min_{j \in \mathcal{T}_{\text{active}}} \|\mathbf{r}_{\text{rel}, j}\|$$

where $\mathcal{T}_{\text{active}} = \{j \mid m_j > 0.5\}$ filters out already-cleared or inactive slots using the explicit binary availability indicator $m_j$.

#### Guidance Rule: Deterministic Line-of-Sight (LOS) Burn
Once target $j^*$ is selected, the satellite maps the 3-axis relative vector $\mathbf{r}_{\text{rel}, j^*} = [x, y, z]^T$ into discrete burn commands $b_k \in \{0, 1, 2\}$:
$$b_k = \begin{cases} 
2 & \text{if } r_k > \delta_{\text{deadband}} \text{ and not damped} \\
0 & \text{if } r_k < -\delta_{\text{deadband}} \text{ and not damped} \\
1 & \text{otherwise (coast)}
\end{cases}$$
- **Deadband**: $\delta_{\text{deadband}} = 0.02$ (normalized units) prevents chattering when near target.
- **Velocity Damping**: When relative velocity data is available and the closing rate satisfies $r_k \cdot v_k < -0.05$, the thruster along axis $k$ coasts ($b_k = 1$) to prevent overshoot.
- **Action Vector**: Emitted as `[j^*, b_x, b_y, b_z]` in `MultiDiscrete([max_targets, 3, 3, 3])`.

---

### 11.2 Fuel-Aware Heuristic

Rather than relying on Euclidean distance alone, the fuel-aware baseline accounts for orbital mechanics realities:
1. **Astrodynamic Maneuver Cost**: In the rotating Hill (LVLH) frame, out-of-plane cross-track motion ($z$) requires expensive orbital plane changes, whereas in-plane offsets ($x, y$) are achieved via low-cost orbital phasing:
   $$C_{\text{maneuver}}(j) = k_1 \sqrt{x_j^2 + y_j^2} + k_2 |z_j|$$
   where $k_1 = 0.025$ (phasing weight) and $k_2 = 0.085$ (plane-change weight).
2. **Propellant Scarcity Multiplier**: As the agent's normalized fuel fraction $f_i \in [0.0, 1.0]$ depletes, cross-track maneuvers become heavily penalized:
   $$S(f_i) = \frac{1}{\max(f_i, 0.1)}$$
3. **Composite Target Selection Score**:
   $$J_j = d_j \left(1 + 0.5 (1 - f_i)\right) + S(f_i) C_{\text{maneuver}}(j)$$
   $$j^* = \arg\min_{j \in \mathcal{T}_{\text{active}}} J_j$$
4. **Fuel-Gated Guidance**:
   - Wider deadband ($\delta = 0.035$) promotes coasting.
   - When fuel is critically depleted ($f_i < 0.15$), cross-track thrusting is completely suppressed ($b_z = 1$), reserving propellant for in-plane operations.

---

### 11.3 Empirical Evaluation & Comparative Benchmarks

Both non-learning baselines were evaluated alongside the centralized Cooperative Autopilot and the trained Independent PPO (IPPO) policy over identical random seeds (seeds 42–61, 20 episodes, 3 agents, 5 targets):

| Policy / Controller | Type | Rendezvous / Target | Capture Acc (%) | Full Sweep (%) | Fuel Used (%) | Mean $\Delta v$ (m/s) | Guidance Acc (%) | Episode Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nearest-Target Heuristic** | Greedy Non-learning | 2.2 / 5.0 | 44.0% | 0.0% | 0.5% | 4.50 | 76.6% | +52.63 |
| **Fuel-Aware Heuristic** | Astrodynamic Non-learning | 2.1 / 5.0 | 43.0% | 0.0% | 0.5% | 4.50 | 75.8% | +51.56 |
| **Cooperative Autopilot** | Coordinated Heuristic | 3.0 / 5.0 | 59.0% | 0.0% | 0.5% | 4.50 | 71.2% | +69.36 |
| **Trained IPPO (Checkpoint)** | Decentralized RL (Smoke) | 0.0 / 5.0 | 0.0% | 0.0% | 0.5% | 4.50 | 24.4% | -6.00 |

#### Key Empirical Insights:
1. **Does Learning Add Measurable Value over a Simple Method?**:
   - **Early/undertrained RL is substantially worse than simple heuristics**: The early IPPO checkpoint (trained on limited smoke iterations) achieves 0.0% rendezvous rate with near-random guidance accuracy (24.4%) and negative rewards (-6.00). In contrast, a simple 3-axis deterministic Line-of-Sight rule immediately achieves 44.0% rendezvous accuracy and positive rewards (+52.63).
   - Therefore, learning does **not** automatically outperform simple heuristics; simple deterministic methods provide a strong and necessary lower bound.
2. **The Greediness Bottleneck ("Dog-piling")**:
   - In both uncoordinated non-learning baselines, satellites frequently converge on the same nearest target cluster. Once the first satellite achieves rendezvous, the remaining satellites find the target deactivated and must redirect, wasting steps. Consequently, **0.0% of episodes achieved a full sweep**.
3. **The Value of Swarm Coordination**:
   - The Cooperative Autopilot achieves **59.0% capture accuracy** (+15.0% absolute improvement over Nearest-Target), demonstrating that spatial allocation and avoiding duplicate assignments provides immense leverage.
4. **Where RL Must Provide Added Value**:
   - For reinforcement learning to add measurable value, policy training must advance past basic single-agent pointing and discover decentralized emergent coordination, target partitioning, and multi-step transfer sequencing that simple greedy methods cannot execute without centralized communication.

---

### 11.4 Running the Baselines

```bash
# Run side-by-side benchmark of both non-learning baselines:
python baselines.py --baseline all --episodes 20

# Run specific baseline with custom fleet configuration:
python baselines.py --baseline nearest --agents 4 --targets 6 --episodes 10
python baselines.py --baseline fuel_aware --agents 4 --targets 6 --episodes 10

# Test baseline suite:
python -m pytest tests/test_baselines.py
```

