Vapor-compression heat pump model with realistic COP as a function of
source and sink temperatures.

Physics basis:
  - Ideal (Carnot) COP:  COP_ideal = T_sink / (T_sink - T_source)   [heating]
  - Real COP:            COP_real  = η * COP_ideal   (η ≈ 0.40–0.55)
  - Second-law efficiency η accounts for compressor irreversibilities,
    heat exchanger temperature approach, and refrigerant non-idealities.

Operating modes:
  HEATING  — heat pump extracts from outdoor air, delivers to zone
  COOLING  — heat pump extracts from zone, rejects to outdoor air
  OFF      — no operation

Defrost:
  Below T_source ~ -5°C, outdoor coil frost degrades performance.
  A simple defrost penalty (COP multiplier) is applied.

Author: HVAC Control Project
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


class HeatPumpMode(Enum):
    OFF     = auto()
    HEATING = auto()
    COOLING = auto()


@dataclass
class HeatPumpSpec:
    """Nameplate / design specifications."""
    capacity_heating_kw: float = 10.0    # rated heating capacity [kW]
    capacity_cooling_kw: float = 9.0     # rated cooling capacity [kW]
    rated_COP_heating:   float = 3.8     # rated COP at 7°C outdoor, 45°C supply
    rated_COP_cooling:   float = 3.5     # rated COP at 35°C outdoor, 7°C evap
    eta_carnot:          float = 0.45    # second-law efficiency
    min_outdoor_heating: float = -20.0   # °C — min outdoor temp for heating
    max_outdoor_cooling: float = 46.0    # °C — max outdoor temp for cooling
    # Supply water / air temperatures
    T_supply_heating:    float = 45.0    # °C (hydronic) or 40°C (forced air)
    T_supply_cooling:    float = 7.0     # °C chilled water / supply air
    # Minimum modulation (inverter-driven, can go down to 20%)
    min_modulation:      float = 0.20


class HeatPump:
    """
    Inverter-driven air-source heat pump model.

    The model computes:
      - COP(T_source, T_sink, modulation)
      - Q_delivered  [W]  — useful heat (heating) or cooling (cooling)
      - P_electric   [W]  — electrical power consumption
      - T_supply     [°C] — supply temperature after heat exchanger
    """

    def __init__(self, spec: HeatPumpSpec = None):
        self.spec = spec or HeatPumpSpec()
        self.mode = HeatPumpMode.OFF
        self.modulation = 0.0       # 0–1 fraction of rated capacity
        self.Q_delivered = 0.0      # W
        self.P_electric  = 0.0      # W
        self.COP         = 0.0
        self.T_supply_out = 20.0    # °C

        # Cumulative energy [Wh]
        self.E_heating = 0.0
        self.E_cooling = 0.0
        self.E_electric = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # COP calculation
    # ─────────────────────────────────────────────────────────────────────

    def cop_heating(self, T_outdoor: float, modulation: float = 1.0) -> float:
        """
        Heating COP as function of outdoor temperature.
        Uses Carnot-based model with:
          - defrost penalty below -5°C
          - part-load efficiency correction
        """
        if T_outdoor < self.spec.min_outdoor_heating:
            return 0.0

        T_src  = T_outdoor + 273.15          # K (source = outdoor air)
        T_sink = self.spec.T_supply_heating + 273.15  # K

        cop_carnot = T_sink / max(T_sink - T_src, 1.0)
        cop_real   = self.spec.eta_carnot * cop_carnot

        # Defrost penalty (linear ramp -10°C → -5°C: 15% → 0% penalty)
        if T_outdoor < -5.0:
            defrost_penalty = 0.15 * min(1.0, (-5.0 - T_outdoor) / 5.0)
            cop_real *= (1.0 - defrost_penalty)

        # Part-load correction (COP peaks near 60–70% modulation)
        cop_real *= self._part_load_factor(modulation)

        return max(cop_real, 1.0)  # COP always ≥ 1 in heating

    def cop_cooling(self, T_outdoor: float, modulation: float = 1.0) -> float:
        """
        Cooling COP as function of outdoor temperature.
        Higher outdoor temp → worse COP (higher condensing pressure).
        """
        if T_outdoor > self.spec.max_outdoor_cooling:
            return 0.0

        T_evap = self.spec.T_supply_cooling + 273.15  # K
        T_cond = T_outdoor + 10.0 + 273.15            # K (+10 approach temp)

        cop_carnot = T_evap / max(T_cond - T_evap, 1.0)
        cop_real   = self.spec.eta_carnot * cop_carnot
        cop_real  *= self._part_load_factor(modulation)

        return max(cop_real, 0.5)

    @staticmethod
    def _part_load_factor(modulation: float) -> float:
        """
        Part-load efficiency factor.
        Inverter-driven compressor is most efficient at ~65% modulation.
        Polynomial fit to typical manufacturer data.
        """
        m = np.clip(modulation, 0.2, 1.0)
        # PLF = -0.5*m² + 0.65*m + 0.85, normalized to 1.0 at peak
        plf = -0.5 * m**2 + 0.65 * m + 0.85
        return plf / 0.9012  # normalize

    # ─────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────

    def step(
        self,
        mode: HeatPumpMode,
        modulation: float,
        T_outdoor: float,
        dt: float,
    ) -> dict:
        """
        Advance heat pump one timestep.

        Parameters
        ----------
        mode        : HeatPumpMode.HEATING / COOLING / OFF
        modulation  : 0–1 fraction of rated capacity
        T_outdoor   : outdoor dry-bulb [°C]
        dt          : timestep [s]

        Returns
        -------
        dict with Q_delivered [W], P_electric [W], COP, T_supply_out [°C]
        """
        self.mode = mode
        self.modulation = np.clip(modulation, 0.0, 1.0)

        if mode == HeatPumpMode.OFF or self.modulation < 0.01:
            self.Q_delivered  = 0.0
            self.P_electric   = 0.0
            self.COP          = 0.0
            self.T_supply_out = T_outdoor
            return self._result()

        if mode == HeatPumpMode.HEATING:
            self.COP = self.cop_heating(T_outdoor, self.modulation)
            Q_rated  = self.spec.capacity_heating_kw * 1000.0  # W
            self.Q_delivered  =  Q_rated * self.modulation  # +ve = heat to zone
            self.T_supply_out = self.spec.T_supply_heating

        elif mode == HeatPumpMode.COOLING:
            self.COP = self.cop_cooling(T_outdoor, self.modulation)
            Q_rated  = self.spec.capacity_cooling_kw * 1000.0  # W
            self.Q_delivered  = -Q_rated * self.modulation  # -ve = cooling
            self.T_supply_out = self.spec.T_supply_cooling

        # Electric power = |Q| / COP
        self.P_electric = abs(self.Q_delivered) / max(self.COP, 0.1)

        # Accumulate energy
        dt_h = dt / 3600.0
        if mode == HeatPumpMode.HEATING:
            self.E_heating  += self.Q_delivered * dt_h / 1000.0  # kWh
        else:
            self.E_cooling  += abs(self.Q_delivered) * dt_h / 1000.0
        self.E_electric += self.P_electric * dt_h / 1000.0

        return self._result()

    def _result(self) -> dict:
        return {
            "Q_delivered":  self.Q_delivered,
            "P_electric":   self.P_electric,
            "COP":          self.COP,
            "T_supply_out": self.T_supply_out,
            "mode":         self.mode.name,
        }

    def reset(self):
        self.mode = HeatPumpMode.OFF
        self.modulation = 0.0
        self.Q_delivered = self.P_electric = self.COP = 0.0
        self.E_heating = self.E_cooling = self.E_electric = 0.0

    def __repr__(self):
        return (
            f"HeatPump({self.mode.name}, mod={self.modulation:.0%}, "
            f"Q={self.Q_delivered/1000:.2f}kW, COP={self.COP:.2f})"
        )
