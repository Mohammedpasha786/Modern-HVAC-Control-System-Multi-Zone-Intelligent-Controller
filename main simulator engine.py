Main simulation engine for multi-zone HVAC system.

Integrates:
  - Multiple ThermalZone models (coupled via internal walls)
  - HumidityModel per zone
  - AirQualityModel per zone
  - HeatPump
  - FSM supervisor
  - PID / MPC controllers

Simulation loop (15-minute default timestep):
  1. Read weather + occupancy inputs
  2. FSM supervisor selects mode
  3. Controllers compute setpoints → commands
  4. Models advance one step
  5. Record results

Author: HVAC Control Project

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import time as wall_time

# Local imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.thermal_zone import ThermalZone, ZoneGeometry, ConstructionProperties
from models.heat_pump import HeatPump, HeatPumpMode, HeatPumpSpec
from models.humidity_model import HumidityModel, AirQualityModel
from controllers.pid_controller import (
    PIDController, PIDGains, CascadedPID,
    TEMP_OUTER_GAINS, TEMP_INNER_GAINS, RH_GAINS, CO2_GAINS
)
from controllers.fsm_controller import (
    HVACStateMachine, ComfortSetpoints, SensorReadings, HVACMode
)


@dataclass
class WeatherInput:
    """Scalar weather inputs at one timestep."""
    T_outdoor: float     # °C
    I_solar:   float     # W/m² horizontal
    RH_outdoor: float    # fraction
    wind_speed: float    # m/s (for infiltration)
    hour_of_day: float   # 0–24


@dataclass
class ZoneConfig:
    """Configuration for one zone."""
    name: str
    geometry: ZoneGeometry
    construction: ConstructionProperties
    T_init: float = 20.0
    n_occupants_profile: Optional[np.ndarray] = None  # 24-hour occupancy profile
    heat_setpoint: float = 21.0
    cool_setpoint: float = 24.0
    # Adjacent zone indices and conductances
    adjacent: list = field(default_factory=list)   # [(zone_idx, UA_W_per_K), ...]


class HVACSimulator:
    """
    Complete multi-zone HVAC simulation engine.

    Usage:
        sim = HVACSimulator(zones_config, dt=900)
        results = sim.run(weather_series, duration_days=7)
    """

    DEFAULT_VENT_RATE = 0.008   # m³/s per person (ASHRAE 62.1 RP-1260)
    MIN_VENT_RATE     = 0.001   # minimum ventilation (2.5 L/s per person)

    def __init__(
        self,
        zones: list[ZoneConfig],
        dt: float = 900.0,      # timestep [s] = 15 min
        controller: str = "pid",  # "pid" | "mpc"
        hp_spec: HeatPumpSpec = None,
    ):
        self.dt = dt
        self.n_zones = len(zones)
        self.zone_configs = zones

        # ── Build zone models ────────────────────────────────────────────
        self.thermal_zones: list[ThermalZone] = []
        self.humidity_models: list[HumidityModel] = []
        self.aq_models: list[AirQualityModel] = []
        self.fsm_controllers: list[HVACStateMachine] = []
        self.pid_controllers: list[CascadedPID] = []

        for i, zc in enumerate(zones):
            tz = ThermalZone(
                zc.name, zc.geometry, zc.construction, zc.T_init, zone_id=i
            )
            hm = HumidityModel(zc.geometry.volume, omega_init=0.008)
            aq = AirQualityModel(
                zc.geometry.volume, zc.geometry.floor_area, C_CO2_init=500.0
            )
            fsm = HVACStateMachine(
                ComfortSetpoints(T_heat_sp=zc.heat_setpoint, T_cool_sp=zc.cool_setpoint)
            )
            pid = CascadedPID(TEMP_OUTER_GAINS, TEMP_INNER_GAINS)

            self.thermal_zones.append(tz)
            self.humidity_models.append(hm)
            self.aq_models.append(aq)
            self.fsm_controllers.append(fsm)
            self.pid_controllers.append(pid)

        # ── Heat pump (shared plant) ──────────────────────────────────────
        self.heat_pump = HeatPump(hp_spec or HeatPumpSpec())

        # ── Results storage ──────────────────────────────────────────────
        self.results: dict[str, list] = self._init_results()
        self.sim_time = 0.0
        self.step_count = 0

    def _init_results(self) -> dict:
        """Initialize results dictionary."""
        cols = ["time_s", "hour_of_day", "T_outdoor", "I_solar",
                "P_electric_kW", "COP", "hp_mode"]
        for zc in self.zone_configs:
            n = zc.name
            cols += [f"{n}_T_zone", f"{n}_T_wall", f"{n}_RH",
                     f"{n}_CO2", f"{n}_Q_hvac_kW", f"{n}_mode"]
        return {c: [] for c in cols}

    # ─────────────────────────────────────────────────────────────────────
    # Simulation step
    # ─────────────────────────────────────────────────────────────────────

    def step(self, weather: WeatherInput, n_occupants: list[float] = None):
        """
        Advance simulation by one timestep (self.dt seconds).

        Parameters
        ----------
        weather     : outdoor conditions at this timestep
        n_occupants : list of occupant count per zone (default: 1 per zone)
        """
        if n_occupants is None:
            n_occupants = [1.0] * self.n_zones

        t = self.sim_time
        total_P_electric = 0.0

        # ── Process each zone ────────────────────────────────────────────
        for i, (tz, hm, aq, fsm, pid, zc) in enumerate(zip(
            self.thermal_zones, self.humidity_models, self.aq_models,
            self.fsm_controllers, self.pid_controllers, self.zone_configs
        )):
            n_occ = n_occupants[i]

            # Current air quality
            omega_outdoor = 0.008  # TODO: compute from RH_outdoor
            omega_zone, RH_zone = hm.omega, hm.omega  # approximation

            # Sensor readings for FSM
            sensors = SensorReadings(
                T_zone=tz.T_zone,
                RH_zone=hm.RH_from_humidity_ratio(tz.T_zone) if hasattr(hm, 'RH_from_humidity_ratio') else 0.5,
                CO2_ppm=aq.C_CO2,
                VOC_ug_m3=aq.C_VOC,
                T_outdoor=weather.T_outdoor,
            )
            # Quick RH calc
            from models.humidity_model import RH_from_humidity_ratio
            RH_cur = RH_from_humidity_ratio(tz.T_zone, hm.omega)
            sensors.RH_zone = RH_cur

            # FSM mode decision
            mode = fsm.update(sensors, sim_time_s=t)
            cmds = fsm.get_commands(mode)

            # PID temperature control → HVAC modulation
            # Outer loop updates every step (simplification)
            hp_mode_str = cmds["hp_mode"]
            if hp_mode_str == "HEATING":
                _, modulation = pid.step(
                    T_zone=tz.T_zone,
                    T_zone_sp=zc.heat_setpoint,
                    T_supply_actual=self.heat_pump.T_supply_out,
                    dt_outer=self.dt, dt_inner=self.dt,
                )
                hp_mode = HeatPumpMode.HEATING
            elif hp_mode_str == "COOLING":
                _, modulation = pid.step(
                    T_zone=tz.T_zone,
                    T_zone_sp=zc.cool_setpoint,
                    T_supply_actual=self.heat_pump.T_supply_out,
                    dt_outer=self.dt, dt_inner=self.dt,
                )
                hp_mode = HeatPumpMode.COOLING
            else:
                modulation = 0.0
                hp_mode = HeatPumpMode.OFF

            # Heat pump step
            hp_result = self.heat_pump.step(
                hp_mode, modulation, weather.T_outdoor, self.dt
            )
            Q_hvac_W = hp_result["Q_delivered"] / max(1, self.n_zones)
            total_P_electric += hp_result["P_electric"]

            # Thermal zone step
            # Adjacent zone coupling
            adj_T = None
            adj_UA = 0.0
            if zc.adjacent:
                adj_idx, adj_ua = zc.adjacent[0]
                adj_T  = self.thermal_zones[adj_idx].T_zone
                adj_UA = adj_ua

            tz.step(
                dt=self.dt, Q_hvac=Q_hvac_W,
                T_outdoor=weather.T_outdoor, I_solar=weather.I_solar,
                n_occupants=n_occ, T_adjacent=adj_T, UA_adjacent=adj_UA,
            )

            # Ventilation flow rate
            vent_frac = cmds["vent_rate_frac"]
            Q_vent_m3_s = vent_frac * self.DEFAULT_VENT_RATE * max(n_occ, 1)
            mdot_vent   = Q_vent_m3_s * 1.20  # kg/s

            # Infiltration
            n_inf = zc.construction.ACH50 * 0.07
            mdot_inf = n_inf * tz.geo.volume / 3600 * 1.20

            # Humidity step
            hm.step(
                dt=self.dt, T_zone=tz.T_zone,
                omega_outdoor=omega_outdoor,
                omega_supply=omega_outdoor,   # simplified: no ERV
                mdot_ventilation=mdot_vent,
                mdot_infiltration=mdot_inf,
                n_occupants=n_occ,
                humidifier_cmd=cmds["humidifier"],
                dehumidifier_cmd=cmds["dehumidifier"],
            )

            # Air quality step
            aq.step(
                dt=self.dt, n_occupants=n_occ,
                Q_ventilation_m3_s=Q_vent_m3_s,
                Q_recirculation_m3_s=cmds["recirculation"] * tz.geo.volume / 3600,
            )

            # Store zone results
            n = zc.name
            self.results[f"{n}_T_zone"].append(tz.T_zone)
            self.results[f"{n}_T_wall"].append(tz.T_wall)
            self.results[f"{n}_RH"].append(RH_cur)
            self.results[f"{n}_CO2"].append(aq.C_CO2)
            self.results[f"{n}_Q_hvac_kW"].append(Q_hvac_W / 1000)
            self.results[f"{n}_mode"].append(mode.name)

        # ── Global results ───────────────────────────────────────────────
        self.results["time_s"].append(t)
        self.results["hour_of_day"].append(weather.hour_of_day)
        self.results["T_outdoor"].append(weather.T_outdoor)
        self.results["I_solar"].append(weather.I_solar)
        self.results["P_electric_kW"].append(total_P_electric / 1000)
        self.results["COP"].append(self.heat_pump.COP)
        self.results["hp_mode"].append(self.heat_pump.mode.name)

        self.sim_time += self.dt
        self.step_count += 1

    # ─────────────────────────────────────────────────────────────────────
    # Run a series
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        weather_series: list[WeatherInput],
        occupancy_series: Optional[list[list[float]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run simulation over a weather time series.

        Parameters
        ----------
        weather_series   : list of WeatherInput, one per timestep
        occupancy_series : list of [n_occ per zone], one per timestep
        verbose          : print progress

        Returns
        -------
        DataFrame with all results
        """
        n_steps = len(weather_series)
        t_start = wall_time.time()

        for k, weather in enumerate(weather_series):
            n_occ = occupancy_series[k] if occupancy_series else None
            self.step(weather, n_occ)

            if verbose and k % max(1, n_steps // 10) == 0:
                elapsed = wall_time.time() - t_start
                print(f"  [{k+1}/{n_steps}] t={self.sim_time/3600:.1f}h  "
                      f"T_zone={self.thermal_zones[0].T_zone:.1f}°C  "
                      f"HP={self.heat_pump.mode.name}  "
                      f"({elapsed:.1f}s elapsed)")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results dict to a tidy DataFrame."""
        df = pd.DataFrame(self.results)
        df["time_h"] = df["time_s"] / 3600.0
        df["time_day"] = df["time_h"] / 24.0
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Energy summary
    # ─────────────────────────────────────────────────────────────────────

    def energy_summary(self, df: pd.DataFrame = None) -> dict:
        """Compute energy consumption summary in kWh."""
        if df is None:
            df = self.to_dataframe()

        dt_h = self.dt / 3600.0
        total_electric = df["P_electric_kW"].sum() * dt_h
        heating_steps = df[df["hp_mode"] == "HEATING"]["P_electric_kW"].sum() * dt_h
        cooling_steps = df[df["hp_mode"] == "COOLING"]["P_electric_kW"].sum() * dt_h

        return {
            "total_electric_kWh": total_electric,
            "heating_electric_kWh": heating_steps,
            "cooling_electric_kWh": cooling_steps,
            "peak_demand_kW": df["P_electric_kW"].max(),
            "avg_COP": df[df["COP"] > 0]["COP"].mean(),
            "simulation_hours": self.sim_time / 3600.0,
        }

    def reset(self):
        """Reset simulator for a new run."""
        for tz in self.thermal_zones:
            tz.reset()
        self.heat_pump.reset()
        self.results = self._init_results()
        self.sim_time = 0.0
        self.step_count = 0


# ─── Weather generators ─────────────────────────────────────────────────────

def synthetic_winter_week(dt: float = 900.0) -> list[WeatherInput]:
    """Generate a synthetic cold winter week in Boston."""
    n = int(7 * 24 * 3600 / dt)
    inputs = []
    for k in range(n):
        t_h = k * dt / 3600.0
        hour = t_h % 24
        day  = int(t_h / 24)
        # Temperature: -5 ± 7°C with daily cycle
        T_out = -5.0 + 7.0 * np.sin(2 * np.pi * (hour - 14) / 24) + np.random.normal(0, 0.5)
        # Solar: daylight hours only
        I_sol = max(0.0, 200 * np.sin(np.pi * (hour - 7) / 10)) if 7 < hour < 17 else 0.0
        inputs.append(WeatherInput(
            T_outdoor=T_out, I_solar=I_sol,
            RH_outdoor=0.75, wind_speed=4.0,
            hour_of_day=hour,
        ))
    return inputs


def synthetic_summer_week(dt: float = 900.0) -> list[WeatherInput]:
    """Generate a synthetic hot summer week."""
    n = int(7 * 24 * 3600 / dt)
    inputs = []
    for k in range(n):
        t_h = k * dt / 3600.0
        hour = t_h % 24
        T_out = 30.0 + 6.0 * np.sin(2 * np.pi * (hour - 15) / 24) + np.random.normal(0, 0.5)
        I_sol = max(0.0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 < hour < 18 else 0.0
        inputs.append(WeatherInput(
            T_outdoor=T_out, I_solar=I_sol,
            RH_outdoor=0.55, wind_speed=2.5,
            hour_of_day=hour,
        ))
    return inputs
