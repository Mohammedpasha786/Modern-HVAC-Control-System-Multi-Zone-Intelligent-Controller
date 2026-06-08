# Modern HVAC Control System — Multi-Zone Intelligent Controller

[![CI/CD](https://github.com/your-org/hvac-control/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/hvac-control/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](tests/)

 *A physics-based, multi-zone HVAC simulation and intelligent control framework** — modeling thermal dynamics, humidity, air quality (CO₂/VOC), and pressure with cascaded PID + Model Predictive Control (MPC) and occupancy-driven optimization.

## Motivation

Modern buildings account for ~40% of global energy consumption. Achieving occupant comfort while minimizing energy use requires controlling:

| Parameter | Target Range | Control Challenge |
|-----------|-------------|-------------------|
| Temperature | 20–24 °C | Thermal mass lag, multi-zone coupling |
| Relative Humidity | 40–60 % RH | Latent heat load, condensation risk |
| CO₂ | < 1000 ppm | Demand-controlled ventilation (DCV) |
| VOC / IAQ | < 500 µg/m³ | Sensor placement, source variability |
| Pressure | ±5 Pa (wrt outside) | Infiltration, stack effect |
| Energy (COP) | > 3.5 | Competing comfort constraints |

This project delivers a **simulation-first** design workflow: model → simulate → optimize → deploy.


## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     HVAC Control Framework                       │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Weather  │───▶│  Disturbance │    │   Supervisor (MPC)  │   │
│  │  Model   │    │  Estimator   │    │  Energy Optimizer   │   │
│  └──────────┘    └──────┬───────┘    └──────────┬──────────┘   │
│                         │                        │               │
│  ┌──────────────────────▼────────────────────────▼──────────┐  │
│  │                   Zone Models (N zones)                    │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ Thermal │  │ Humidity │  │   IAQ    │  │ Pressure │  │  │
│  │  │  (RC)   │  │  (mass   │  │  (CO₂+  │  │  (flow   │  │  │
│  │  │  model  │  │ balance) │  │   VOC)   │  │  model)  │  │  │
│  │  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │  │
│  └───────┼────────────┼─────────────┼──────────────┼─────────┘  │
│          ▼            ▼             ▼              ▼             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Cascaded PID Controllers (per zone)            │    │
│  │  Temp PID → Valve  |  RH PID → Humidifier  |  CO₂ DCV  │    │
│  └─────────────────────────────────────────────────────────┘    │
│          │                                                        │
│  ┌───────▼──────────────────────────────────────────────────┐   │
│  │               HVAC Plant (Heat Pump + AHU + ERV)          │   │
│  │  Heat Pump (COP~3-5)  |  Air Handler  |  Energy Recovery  │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
## Features

### Physics-Based Models
- **Thermal RC network** — walls, windows, roof, floor with realistic U-values and thermal mass
- **Humidity mass balance** — latent heat, infiltration moisture, occupant generation
- **CO₂ / IAQ model** — occupant generation (0.3 L/min/person), DCV ventilation
- **Pressure network** — fan curves, duct resistance, stack effect
- **Heat pump COP model** — COP(T_source, T_sink) using refrigerant cycle approximation

### Control Strategies
- **Cascaded PID** — inner loop (supply air temp) + outer loop (zone temp)
- **Model Predictive Control (MPC)** — 24-hour horizon, weather preview, occupancy schedule
- **Stateflow-equivalent FSM** — seasonal mode switching (Heat / Cool / Ventilate / Standby)
- **Demand-Controlled Ventilation (DCV)** — CO₂-driven fresh air rate
- **Energy Recovery Ventilator (ERV)** — heat and moisture exchange between exhaust/supply

### Analysis & Reporting
- Time-series simulation (15-min timestep, 1-year capability)
- Energy breakdown (heating / cooling / fan / humidification)
- Comfort metrics: PMV, PPD, ASHRAE 55
- Automatic PDF/HTML report generation

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/hvac-control.git
cd hvac-control

# 2. Install
pip install -e ".[dev]"

# 3. Run a basic single-zone simulation (24 hours)
python -m hvac_sim.run --config configs/single_zone_winter.yaml --plot

# 4. Run multi-zone house (5 zones, 1 week)
python -m hvac_sim.run --config configs/multi_zone_house.yaml --days 7 --plot

# 5. Launch interactive dashboard
streamlit run dashboard/app.py

---

## Models

### Thermal Zone (RC Network)

Each zone is modeled as a 2nd-order RC network:

```
  T_outside ──[R_wall]──┬── T_wall_inner ──[R_inner]──┬── T_zone
                        │                              │
                       C_wall                        C_air
                        │                              │
                        └──────────────────────────────┘
                                        │
                              Q_hvac + Q_solar + Q_internal
```

State equations:
```
C_air  · dT_zone/dt  = (T_wall - T_zone)/R_inner + Q_hvac + Q_solar + Q_occ + Q_infil
C_wall · dT_wall/dt  = (T_out - T_wall)/R_wall   - (T_wall - T_zone)/R_inner
```

### Heat Pump COP Model

```
COP_heating = η_carnot · T_supply / (T_supply - T_source)   [η_carnot ~ 0.45]
COP_cooling = η_carnot · T_evap   / (T_cond - T_evap)
```

### Energy Recovery Ventilator

```
ε_sensible  = (T_supply_out - T_outdoor) / (T_exhaust_in - T_outdoor)   [~ 0.75]
ε_latent    = (ω_supply_out - ω_outdoor) / (ω_exhaust_in - ω_outdoor)   [~ 0.65]
```



## Controllers

### Finite State Machine (Mode Logic)

        T < T_heat_sp - δ           T > T_cool_sp + δ
   ┌──────────────────────┐    ┌───────────────────────┐
   │                      ▼    │                       ▼
[STANDBY] ──────────▶ [HEATING] ◀──────────────▶ [COOLING]
   │  ▲                   │                           │
   │  └───────────────────┘                           │
   │         T in range                               │
   ▼                                                  │
[VENTILATE] ◀─── CO₂ > 800 ppm ────────────────────▶│
   │                                                  │
   └─────────── CO₂ < 600 ppm ──────────────────────▶┘
```

### MPC Formulation

min  Σ [ w_T·(T_zone - T_sp)² + w_u·ΔQ² + w_e·P_elec ]
 u
s.t. x(k+1) = A·x(k) + B·u(k) + E·d(k)
     T_min ≤ T_zone ≤ T_max
     0 ≤ Q_hvac ≤ Q_max
     |ΔQ| ≤ ΔQ_max
```
Solved with OSQP at each timestep over a 24-step (6h) horizon.


## Simulation Results

| Scenario | Heating Energy | Cooling Energy | Avg Comfort (PMV) | Peak Demand |
|----------|---------------|----------------|-------------------|-------------|
| PID only (winter week) | 312 kWh | — | -0.42 | 8.2 kW |
| MPC (winter week) | **267 kWh** | — | **-0.18** | **6.1 kW** |
| PID only (summer week) | — | 198 kWh | +0.51 | 6.8 kW |
| MPC (summer week) | — | **164 kWh** | **+0.21** | **5.2 kW** |

MPC achieves **~15% energy reduction** and **improved comfort** vs. baseline PID.


## Energy Analysis

Annual simulation (5-zone house, Boston TMY3 climate):
Annual Energy Breakdown:
  Space Heating (heat pump):   4,820 kWh  (48%)
  Space Cooling (heat pump):   2,140 kWh  (21%)
  Ventilation fans:              680 kWh   (7%)
  Humidification:                420 kWh   (4%)
  Auxiliary (pumps, controls):   340 kWh   (3%)
  ─────────────────────────────────────────
  Total HVAC:                  8,400 kWh/yr
  Without ERV:                10,200 kWh/yr  (+21%)
  Without MPC:                 9,600 kWh/yr  (+14%)
```

## Roadmap

- [x] Single-zone thermal + PID
- [x] Humidity and CO₂ models
- [x] Heat pump COP model
- [x] ERV effectiveness model
- [x] Multi-zone coupling
- [x] MPC controller
- [x] FSM mode switching
- [x] Annual simulation + energy reporting
- [ ] Reinforcement learning controller (PPO)
- [ ] FMI/FMU export for co-simulation
- [ ] Hardware-in-the-loop (HIL) test bench
- [ ] BACnet/Modbus integration for real deployment
- [ ] Digital twin dashboard (live sensor fusion)

## References

1. ASHRAE Fundamentals Handbook (2021)
2. Crawley et al., "EnergyPlus: Creating a new-generation building energy simulation program," *Energy and Buildings*, 2001
3. Afram & Janabi-Sharifi, "Theory and applications of HVAC control systems," *Building and Environment*, 2014
4. Privara et al., "Model predictive control of a building heating system," *Energy and Buildings*, 2011
5. Sturzebecher et al., "Thermal Dynamic Modeling of Multi-Zone Buildings in MATLAB/Simulink," 2020

## License

MIT © 2024 HVAC Control Project Contributors
