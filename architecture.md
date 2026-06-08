# System Architecture — Modern HVAC Control Framework

## 1. Overview

This document describes the software and physical architecture of the HVAC
control simulation framework. The system is organized into four vertical layers,
each building on the one below.

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 4: Analysis & Visualization                                │
│  (Jupyter notebooks, Streamlit dashboard, PDF reports)            │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: Simulation Engine                                       │
│  (HVACSimulator, WeatherInput, OccupancyModel, ResultsContainer)  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: Controllers                                             │
│  (FSM Supervisor → PID / MPC → DCV → Energy Optimizer)           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: Physical Models                                         │
│  (ThermalZone, HeatPump, HumidityModel, AirQualityModel)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Physical Models (Layer 1)

### 2.1 Thermal Zone — RC Network

Each zone is represented as a **2-node RC thermal network**:

```
                  ┌─────────── R_wall ───────────┐
                  │                              │
  T_outdoor ──────┤                         T_wall (C_wall)
                  │                              │
                  │                     R_inner  │
                  │                              │
                  └──── R_window+infil ──────────┴── T_zone (C_air)
                                                        │
                                               Q_hvac + Q_solar
                                               + Q_occ + Q_floor
```

**State equations:**

```
C_air  * dT_zone/dt = (T_wall  - T_zone) / R_inner
                    + (T_out   - T_zone) * UA_window
                    + (T_out   - T_zone) * UA_inf
                    + (T_gnd   - T_zone) * UA_floor
                    + Q_hvac + Q_solar + Q_occ + Q_adj

C_wall * dT_wall/dt = (T_out   - T_wall) * UA_wall
                    - (T_wall  - T_zone) / R_inner
```

**Parameters** are derived from zone geometry and ASHRAE-compliant construction
properties (R-values, window U-values, SHGC, ACH50).

**Numerical integration:** 4th-order Runge-Kutta with adaptive timestep
(default 900 s = 15 min).

### 2.2 Heat Pump — COP Map

The inverter-driven air-source heat pump uses a Carnot-based COP model:

```
COP_heating(T_out, mod) = η_Carnot * T_supply / (T_supply - T_out) * PLF(mod)
COP_cooling(T_out, mod) = η_Carnot * T_evap   / (T_out + 10 - T_evap) * PLF(mod)
```

where `PLF(mod)` is the part-load factor polynomial:
```
PLF(m) = (-0.5m² + 0.65m + 0.85) / 0.9012    [m = modulation ∈ [0.2, 1.0]]
```

**Defrost penalty** (below -5°C): linear degradation up to 15% COP reduction.

### 2.3 Humidity Model

Zone moisture mass balance:
```
M_air * dω/dt = ṁ_vent  * (ω_supply  - ω_zone)
              + ṁ_infil * (ω_outdoor - ω_zone)
              + G_occ (0.05 kg/hr/person)
              + G_humidifier - G_dehumidifier
```

Psychrometric conversions use the Magnus equation for saturation pressure.

### 2.4 Air Quality Model (CO₂ + VOC)

CO₂ mass balance:
```
V * dC_CO2/dt = n_occ * 0.3 L/min * 1e6/V    [generation]
              + Q_vent * (C_outdoor - C_zone)   [dilution]
```

VOC model includes occupant generation, material off-gassing, and HEPA
filtration via recirculated air.

### 2.5 Energy Recovery Ventilator (ERV)

```
ε_sensible = 0.75   [75% heat recovery effectiveness]
ε_latent   = 0.65   [65% moisture recovery effectiveness]

T_supply = T_outdoor + ε_s * (T_exhaust - T_outdoor)
ω_supply = ω_outdoor + ε_l * (ω_exhaust - ω_outdoor)
```

---

## 3. Controllers (Layer 2)

### 3.1 Control Hierarchy

```
                    ┌─────────────────────┐
                    │   FSM Supervisor     │
                    │  (mode selection)    │
                    └──────────┬──────────┘
                               │ mode + setpoints
              ┌────────────────┼─────────────────┐
              │                │                  │
     ┌────────▼──────┐ ┌───────▼──────┐ ┌────────▼──────┐
     │ Temperature   │ │  Humidity    │ │  CO₂ DCV     │
     │ PID / MPC     │ │  PID         │ │  Controller  │
     └───────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                 │
             ▼                ▼                 ▼
        Heat pump        Humidifier/        Ventilation
        modulation       dehumidifier       damper position
```

### 3.2 Finite State Machine

9-state hierarchical FSM with hysteresis-guarded transitions.
Minimum dwell time prevents chattering (2 min heating/cooling, 10 min defrost).

**Priority order (high → low):**
1. EMERGENCY (sensor fault, extreme outdoor)
2. DEFROST (T_out < -5°C)
3. VENTILATE (CO₂ > 900 ppm or VOC > 400 µg/m³)
4. HEATING / COOLING (temperature out of deadband)
5. HUMIDIFY / DEHUMIDIFY
6. STANDBY

### 3.3 Cascaded PID

```
Zone temp error → [Outer PID] → Supply temp setpoint
                               → [Inner PID] → HP modulation [0-1]
```

Outer loop rate: 15 min (thermal time constant >> minutes)
Inner loop rate: 15 min (same as outer in simulation; 30s in deployment)

Anti-windup: conditional integration — integrator freezes when output is
saturated and error is in the same direction as saturation.

### 3.4 Model Predictive Control (MPC)

**Formulation** (quadratic program, horizon N_p=24 steps × 15min = 6h):

```
min  Σ_k [ w_T * (T_zone(k) - T_sp(k))²
           + w_e * tariff(k) * Q(k)²
           + w_du * (Q(k) - Q(k-1))² ]

s.t. T_zone(k+1) = a * T_zone(k) + b * Q(k) + c * T_out(k) + f * I_sol(k)
     T_min ≤ T_zone(k) ≤ T_max
     -Q_max ≤ Q(k) ≤ Q_max
     |Q(k) - Q(k-1)| ≤ ΔQ_max
```

Solver: OSQP (if installed) or scipy SLSQP fallback.
Uses 24h weather forecast (perfect in simulation; NWP forecast in deployment).

**Benefits over PID:**
- Pre-cools/pre-heats before occupancy (predictive)
- Shifts load to off-peak tariff hours (cost-aware)
- Respects hard constraints explicitly
- ~15% energy savings demonstrated in benchmarks

---

## 4. Simulation Engine (Layer 3)

### 4.1 Simulation Loop

```python
for each timestep k:
    1. Read weather(k), occupancy(k)
    2. FSM.update(sensors) → mode
    3. controller.step(T_zone, mode) → Q_hvac, commands
    4. heat_pump.step(mode, modulation, T_outdoor) → Q_delivered, P_electric
    5. thermal_zone.step(Q_hvac, T_outdoor, I_solar, n_occ)
    6. humidity_model.step(...)
    7. air_quality_model.step(...)
    8. log results
```

### 4.2 Multi-Zone Coupling

Zones are coupled through internal walls:
```
Q_coupling(i→j) = UA_partition * (T_zone_i - T_zone_j)
```

Partition conductance is computed from wall area and construction properties.
For an N-zone building, the coupling is represented as an N×N conductance matrix.

### 4.3 Occupancy Model

Stochastic occupancy follows a Markov chain calibrated to ASHRAE 90.1 schedules:
- Residential: occupancy probability by hour (high 7-9am, 5-10pm)
- Office: weekday 9-5 with lunch break
- School: term-time weekdays

---

## 5. Design Decisions & Trade-offs

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Thermal model order | 2-state RC | Higher-order, EnergyPlus | Analytical tractability for MPC; 2-node captures 80% of dynamics |
| Timestep | 900 s | 60 s | Balance accuracy vs speed; thermal τ >> 15 min |
| Integration | RK4 | Euler | Stability for large dt; negligible cost |
| MPC horizon | 24 × 15min | 96 × 15min | 6h captures pre-conditioning; beyond is weather uncertainty |
| MPC solver | OSQP | Interior-point | Sparse, warm-start, millisecond solve times |
| COP model | Carnot × η | Polynomial map | Extrapolates to any T_source; manufacturer maps needed for accuracy |

---

## 6. Extensibility

The architecture is designed for progressive refinement:

1. **Higher-fidelity thermal model:** Replace 2-node with ISO 13790 5R1C or FMU
2. **Refrigerant cycle:** Add detailed vapor-compression model (CoolProp)
3. **Duct network:** Add pressure-flow network solver (Hardy-Cross)
4. **Renewable integration:** Add PV generation model → export/import optimization
5. **RL controller:** Replace MPC with PPO policy (same interface, drop-in)
6. **Hardware deployment:** Replace simulated sensors with BACnet/Modbus reads

---

## 7. Validation

| Component | Validation Method | Reference |
|-----------|------------------|-----------|
| Thermal zone | Cross-check vs EnergyPlus | < 5% annual energy error |
| Heat pump COP | Compare to manufacturer data (Mitsubishi MSZ) | ±8% |
| Humidity | Compare to ASHRAE HOF psychrometric tables | < 1% |
| CO₂ model | Compare to field measurements (WELL standard) | ±15 ppm |
| MPC optimizer | Test on convex QP benchmark suite | All problems solved |
