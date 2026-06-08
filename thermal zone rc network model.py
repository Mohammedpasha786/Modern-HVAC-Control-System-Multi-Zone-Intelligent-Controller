thermal_zone.py
===============
Physics-based thermal zone model using a 2nd-order RC (resistor-capacitor)
network analogous to an electrical circuit.

Thermal Analogy:
  Temperature  ↔  Voltage
  Heat flow    ↔  Current
  Thermal resistance R  ↔  Electrical resistance
  Thermal capacitance C ↔  Electrical capacitance

State equations (2-node model):
  C_air  * dT_zone/dt = (T_wall - T_zone)/R_inner
                        + Q_hvac + Q_solar + Q_occupants + Q_infiltration
  C_wall * dT_wall/dt = (T_outdoor - T_wall)/R_wall
                        - (T_wall - T_zone)/R_inner

Author: HVAC Control Project
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ZoneGeometry:
    """Physical dimensions and construction of a zone."""
    floor_area: float         # m²
    ceiling_height: float     # m  (default 2.7)
    wall_area: float          # m² (total opaque wall area)
    window_area: float        # m² (total glazing area)
    roof_area: float          # m² (only for top-floor zones)
    floor_slab: bool = True   # True if concrete slab (high thermal mass)

    @property
    def volume(self) -> float:
        return self.floor_area * self.ceiling_height


@dataclass
class ConstructionProperties:
    """
    Thermal properties of building envelope.
    All R-values in [m²·K/W] (SI), U-values in [W/m²·K].
    """
    # Walls
    R_wall: float = 3.5          # m²K/W  (~R-20 imperial)
    # Windows (double-pane low-e)
    U_window: float = 1.8        # W/m²K
    # Roof
    R_roof: float = 5.3          # m²K/W  (~R-30)
    # Floor
    R_floor: float = 2.6         # m²K/W
    # Infiltration (air changes per hour at 50 Pa, typical 3–8 ACH50)
    ACH50: float = 5.0
    # Solar heat gain coefficient of glazing
    SHGC: float = 0.35
    # Thermal mass densities
    wall_density: float = 800.0  # kg/m³ (light frame)
    wall_thickness: float = 0.15 # m
    wall_cp: float = 840.0       # J/(kg·K)
    air_density: float = 1.20    # kg/m³
    air_cp: float = 1006.0       # J/(kg·K)


class ThermalZone:
    """
    2-state thermal model of a single conditioned zone.

    States:
        x[0] = T_zone  [°C]  — zone air temperature
        x[1] = T_wall  [°C]  — effective wall (thermal mass) temperature

    Inputs:
        Q_hvac      [W]  — net heating (+) or cooling (-) from HVAC system
        T_outdoor   [°C] — outdoor dry-bulb temperature
        I_solar     [W/m²] — horizontal solar irradiance
        n_occupants [−]   — number of occupants in zone

    Parameters are derived from ZoneGeometry + ConstructionProperties.
    """

    # Metabolic heat per occupant (seated office activity, ASHRAE 55)
    Q_PER_PERSON_W = 80.0  # W sensible heat
    # Infiltration n50 → nACH conversion (Sherman-Grimsrud simplified)
    N50_TO_NEFF = 0.07     # effective ACH = ACH50 * 0.07

    def __init__(
        self,
        name: str,
        geometry: ZoneGeometry,
        construction: ConstructionProperties,
        T_init: float = 20.0,
        zone_id: int = 0,
    ):
        self.name = name
        self.zone_id = zone_id
        self.geo = geometry
        self.con = construction

        # ── Compute lumped thermal parameters ───────────────────────────
        # Thermal capacitances [J/K]
        self.C_air = (
            geometry.volume
            * construction.air_density
            * construction.air_cp
        )
        self.C_wall = (
            geometry.wall_area
            * construction.wall_density
            * construction.wall_thickness
            * construction.wall_cp
        )
        if geometry.floor_slab:
            # Concrete slab: density 2300 kg/m³, cp 880 J/(kg·K), 0.1 m thick
            self.C_wall += geometry.floor_area * 2300 * 880 * 0.10

        # Thermal conductances [W/K]
        self.UA_wall    = geometry.wall_area   / construction.R_wall
        self.UA_window  = geometry.window_area * construction.U_window
        self.UA_roof    = geometry.roof_area   / construction.R_roof
        self.UA_floor   = geometry.floor_area  / construction.R_floor
        # Inner surface (wall thermal mass ↔ air)
        self.UA_inner   = geometry.wall_area * 8.0  # convective h=8 W/m²K

        # Total envelope conductance (outdoor→zone for windows+infiltration)
        n_inf = construction.ACH50 * self.N50_TO_NEFF  # effective ACH
        self.mdot_inf = n_inf * geometry.volume / 3600 * construction.air_density
        self.UA_inf = self.mdot_inf * construction.air_cp  # [W/K]

        # ── State vector ─────────────────────────────────────────────────
        self.T_zone = T_init
        self.T_wall = T_init  # start at equilibrium
        self.state = np.array([T_init, T_init], dtype=float)

        # ── Logging ──────────────────────────────────────────────────────
        self.history: dict[str, list] = {
            "T_zone": [], "T_wall": [], "Q_hvac": [],
            "Q_solar": [], "Q_occ": [], "Q_loss": [],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Core dynamics
    # ─────────────────────────────────────────────────────────────────────

    def derivatives(
        self,
        state: np.ndarray,
        Q_hvac: float,
        T_outdoor: float,
        I_solar: float,
        n_occupants: float,
        T_adjacent: Optional[float] = None,
        UA_adjacent: float = 0.0,
    ) -> np.ndarray:
        """
        Compute dT/dt for [T_zone, T_wall].

        Parameters
        ----------
        state       : [T_zone, T_wall]
        Q_hvac      : HVAC sensible heat input to zone [W], + = heating
        T_outdoor   : outdoor dry-bulb [°C]
        I_solar     : total horizontal solar irradiance [W/m²]
        n_occupants : number of occupants (fractional OK)
        T_adjacent  : temperature of adjacent zone [°C] (optional)
        UA_adjacent : conductance to adjacent zone [W/K]

        Returns
        -------
        dxdt : [dT_zone/dt, dT_wall/dt]
        """
        T_zone, T_wall = state

        # Solar gains through glazing [W]
        Q_solar = self.geo.window_area * self.con.SHGC * I_solar

        # Occupant heat gains [W]
        Q_occ = n_occupants * self.Q_PER_PERSON_W

        # Infiltration [W]
        Q_infil = self.UA_inf * (T_outdoor - T_zone)

        # Window conduction [W]
        Q_window = self.UA_window * (T_outdoor - T_zone)

        # Roof conduction [W]
        Q_roof = self.UA_roof * (T_outdoor - T_zone)

        # Floor conduction [W] (to ground, approx 10°C year-round)
        Q_floor = self.UA_floor * (10.0 - T_zone)

        # Adjacent zone coupling [W]
        Q_adj = UA_adjacent * (T_adjacent - T_zone) if T_adjacent is not None else 0.0

        # Wall conduction (outdoor → wall thermal mass) [W]
        Q_wall_outer = self.UA_wall * (T_outdoor - T_wall)

        # Wall inner surface convection (wall ↔ zone air) [W]
        Q_wall_inner = self.UA_inner * (T_wall - T_zone)

        # --- Zone air energy balance ---
        dT_zone_dt = (
            Q_wall_inner
            + Q_hvac
            + Q_solar
            + Q_occ
            + Q_infil
            + Q_window
            + Q_roof
            + Q_floor
            + Q_adj
        ) / self.C_air

        # --- Wall (thermal mass) energy balance ---
        dT_wall_dt = (Q_wall_outer - Q_wall_inner) / self.C_wall

        # Store for diagnostics
        self._last_Q = {
            "Q_hvac": Q_hvac,
            "Q_solar": Q_solar,
            "Q_occ": Q_occ,
            "Q_loss": -(Q_infil + Q_window + Q_roof),
        }

        return np.array([dT_zone_dt, dT_wall_dt])

    def step(
        self,
        dt: float,
        Q_hvac: float,
        T_outdoor: float,
        I_solar: float,
        n_occupants: float,
        T_adjacent: Optional[float] = None,
        UA_adjacent: float = 0.0,
        method: str = "rk4",
    ) -> tuple[float, float]:
        """
        Advance state by one timestep dt [s] using RK4 or Euler.

        Returns (T_zone, T_wall) after the step.
        """
        kwargs = dict(
            Q_hvac=Q_hvac, T_outdoor=T_outdoor, I_solar=I_solar,
            n_occupants=n_occupants, T_adjacent=T_adjacent,
            UA_adjacent=UA_adjacent,
        )
        if method == "rk4":
            self.state = self._rk4(self.state, dt, **kwargs)
        else:
            dxdt = self.derivatives(self.state, **kwargs)
            self.state = self.state + dt * dxdt

        self.T_zone, self.T_wall = self.state

        # Log
        self.history["T_zone"].append(self.T_zone)
        self.history["T_wall"].append(self.T_wall)
        for k, v in self._last_Q.items():
            self.history.get(k, []).append(v)

        return self.T_zone, self.T_wall

    def _rk4(self, x: np.ndarray, dt: float, **kwargs) -> np.ndarray:
        """Classic 4th-order Runge-Kutta integration."""
        k1 = self.derivatives(x,          **kwargs)
        k2 = self.derivatives(x + dt/2*k1, **kwargs)
        k3 = self.derivatives(x + dt/2*k2, **kwargs)
        k4 = self.derivatives(x + dt   *k3, **kwargs)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ─────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, T_init: float = 20.0):
        """Reset state and history."""
        self.state = np.array([T_init, T_init])
        self.T_zone, self.T_wall = T_init, T_init
        self.history = {k: [] for k in self.history}

    @property
    def total_UA(self) -> float:
        """Total envelope conductance [W/K] — useful for sanity checks."""
        return self.UA_wall + self.UA_window + self.UA_roof + self.UA_floor + self.UA_inf

    @property
    def time_constant_hours(self) -> float:
        """Approximate zone thermal time constant [hours]."""
        return (self.C_air + self.C_wall) / (self.total_UA * 3600)

    def __repr__(self):
        return (
            f"ThermalZone('{self.name}', "
            f"floor={self.geo.floor_area}m², "
            f"τ={self.time_constant_hours:.1f}h, "
            f"T_zone={self.T_zone:.1f}°C)"
        )


# ─── Factory helpers ─────────────────────────────────────────────────────────

def make_bedroom(name: str = "Bedroom", **kwargs) -> ThermalZone:
    geo = ZoneGeometry(floor_area=15, ceiling_height=2.7,
                       wall_area=40, window_area=3.0, roof_area=0)
    return ThermalZone(name, geo, ConstructionProperties(), **kwargs)


def make_living_room(name: str = "LivingRoom", **kwargs) -> ThermalZone:
    geo = ZoneGeometry(floor_area=30, ceiling_height=2.7,
                       wall_area=60, window_area=8.0, roof_area=0)
    return ThermalZone(name, geo, ConstructionProperties(), **kwargs)


def make_office(name: str = "Office", **kwargs) -> ThermalZone:
    geo = ZoneGeometry(floor_area=20, ceiling_height=2.7,
                       wall_area=45, window_area=4.0, roof_area=0)
    con = ConstructionProperties(ACH50=3.0)   # tighter office envelope
    return ThermalZone(name, geo, con, **kwargs)
