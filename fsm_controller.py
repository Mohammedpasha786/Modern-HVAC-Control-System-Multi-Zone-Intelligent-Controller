"""
fsm_controller.py
=================
Finite State Machine (FSM) controller for HVAC system mode selection.

Implements seasonal and event-driven mode transitions using a Stateflow-
equivalent Python implementation. This is the top-level supervisor that
determines whether the system should be heating, cooling, ventilating,
dehumidifying, or in standby.

States:
  STANDBY      — Zone within all comfort bounds; minimal energy use
  HEATING      — Zone below heating setpoint; heat pump in heating mode
  COOLING      — Zone above cooling setpoint; heat pump in cooling mode
  VENTILATING  — CO₂ or VOC above thresholds; increased fresh air rate
  DEHUMIDIFY   — RH above max; dehumidifier or cooling-based dehumidification
  HUMIDIFY     — RH below min; humidifier active
  DEFROST      — Outdoor temp < -5°C; heat pump in defrost cycle
  EMERGENCY    — Sensor failure or extreme conditions; safe fallback

Transitions are guarded by hysteresis bands to prevent chattering.

Author: HVAC Control Project
"""

from enum import Enum, auto
from dataclasses import dataclass
import time


class HVACMode(Enum):
    STANDBY    = auto()
    HEATING    = auto()
    COOLING    = auto()
    VENTILATE  = auto()
    DEHUMIDIFY = auto()
    HUMIDIFY   = auto()
    DEFROST    = auto()
    EMERGENCY  = auto()


@dataclass
class ComfortSetpoints:
    """Comfort and trigger setpoints with hysteresis bands."""
    # Temperature
    T_heat_sp:   float = 21.0   # °C — heating setpoint
    T_cool_sp:   float = 24.0   # °C — cooling setpoint
    T_heat_dead: float = 0.5    # °C — hysteresis band (don't heat until T < sp - dead)
    T_cool_dead: float = 0.5    # °C

    # Humidity
    RH_min:      float = 0.35   # 35% RH — humidify below this
    RH_max:      float = 0.60   # 60% RH — dehumidify above this
    RH_dead:     float = 0.03   # 3% hysteresis

    # CO₂
    CO2_high:    float = 900.0  # ppm — start ventilating
    CO2_low:     float = 650.0  # ppm — can stop ventilation boost
    CO2_alarm:   float = 1500.0 # ppm — emergency ventilation

    # VOC
    VOC_high:    float = 400.0  # µg/m³
    VOC_low:     float = 200.0  # µg/m³

    # Outdoor temperature limits
    T_defrost:   float = -5.0   # °C — begin defrost protection
    T_emergency: float = -25.0  # °C — outdoor too cold, risk of pipe freeze


@dataclass
class SensorReadings:
    """Current sensor values passed to the FSM."""
    T_zone:      float = 22.0   # °C
    RH_zone:     float = 0.50   # fraction
    CO2_ppm:     float = 600.0  # ppm
    VOC_ug_m3:   float = 50.0   # µg/m³
    T_outdoor:   float = 10.0   # °C
    # Flags
    sensor_fault: bool = False


class HVACStateMachine:
    """
    Hierarchical finite state machine for HVAC mode supervision.

    Transition priority (high → low):
      1. Emergency (sensor fault, extreme outdoor)
      2. Defrost
      3. CO₂ / VOC ventilation (overrides comfort if IAQ critical)
      4. Temperature (heating / cooling)
      5. Humidity (humidify / dehumidify)
      6. Standby

    Mode can be overridden externally (e.g., manual override, schedule).
    """

    def __init__(self, setpoints: ComfortSetpoints = None):
        self.sp = setpoints or ComfortSetpoints()
        self.mode = HVACMode.STANDBY
        self._prev_mode = HVACMode.STANDBY
        self._mode_entry_time = time.monotonic()
        self._mode_duration_s: float = 0.0

        # Minimum dwell time in each mode [s] — prevents rapid switching
        self._min_dwell: dict[HVACMode, float] = {
            HVACMode.HEATING:    120.0,   # 2 min
            HVACMode.COOLING:    120.0,
            HVACMode.VENTILATE:   60.0,
            HVACMode.DEHUMIDIFY: 180.0,
            HVACMode.HUMIDIFY:   180.0,
            HVACMode.DEFROST:    600.0,   # 10 min defrost cycle
            HVACMode.STANDBY:     30.0,
            HVACMode.EMERGENCY:    0.0,
        }

        # Transition history for logging
        self.history: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────
    # Main update
    # ─────────────────────────────────────────────────────────────────────

    def update(self, sensors: SensorReadings, sim_time_s: float = 0.0) -> HVACMode:
        """
        Evaluate all transitions and return the new (or current) mode.

        Parameters
        ----------
        sensors     : current sensor readings
        sim_time_s  : simulation time [s] for logging

        Returns
        -------
        mode : current HVACMode after transition evaluation
        """
        sp = self.sp
        s  = sensors

        # Compute dwell time in current mode
        now = time.monotonic()
        self._mode_duration_s = now - self._mode_entry_time

        # ── Priority 1: Emergency ──────────────────────────────────────
        if s.sensor_fault or s.T_outdoor < sp.T_emergency:
            return self._transition(HVACMode.EMERGENCY, sensors, sim_time_s)

        # ── Priority 2: Defrost ────────────────────────────────────────
        if s.T_outdoor < sp.T_defrost:
            if self.mode != HVACMode.DEFROST:
                return self._transition(HVACMode.DEFROST, sensors, sim_time_s)
        else:
            if self.mode == HVACMode.DEFROST:
                return self._transition(HVACMode.STANDBY, sensors, sim_time_s)

        # ── Priority 3: CO₂ / IAQ ventilation ─────────────────────────
        co2_high = s.CO2_ppm > sp.CO2_high or s.VOC_ug_m3 > sp.VOC_high
        co2_ok   = s.CO2_ppm < sp.CO2_low  and s.VOC_ug_m3 < sp.VOC_low

        if co2_high and self._can_transition():
            return self._transition(HVACMode.VENTILATE, sensors, sim_time_s)

        # ── Priority 4: Temperature ────────────────────────────────────
        need_heat = s.T_zone < sp.T_heat_sp - sp.T_heat_dead
        need_cool = s.T_zone > sp.T_cool_sp + sp.T_cool_dead
        temp_ok   = sp.T_heat_sp - sp.T_heat_dead <= s.T_zone <= sp.T_cool_sp + sp.T_cool_dead

        if self.mode not in (HVACMode.VENTILATE,) or co2_ok:
            if need_heat and self._can_transition():
                return self._transition(HVACMode.HEATING, sensors, sim_time_s)
            if need_cool and self._can_transition():
                return self._transition(HVACMode.COOLING, sensors, sim_time_s)

        # ── Priority 5: Humidity ───────────────────────────────────────
        rh_high = s.RH_zone > sp.RH_max + sp.RH_dead
        rh_low  = s.RH_zone < sp.RH_min - sp.RH_dead

        if rh_high and temp_ok and self._can_transition():
            return self._transition(HVACMode.DEHUMIDIFY, sensors, sim_time_s)
        if rh_low  and temp_ok and self._can_transition():
            return self._transition(HVACMode.HUMIDIFY, sensors, sim_time_s)

        # ── Priority 6: Standby (all conditions met) ───────────────────
        if (temp_ok and co2_ok and not rh_high and not rh_low
                and self._can_transition()):
            return self._transition(HVACMode.STANDBY, sensors, sim_time_s)

        return self.mode

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _can_transition(self) -> bool:
        """True if minimum dwell time in current mode has been exceeded."""
        min_dwell = self._min_dwell.get(self.mode, 60.0)
        return self._mode_duration_s >= min_dwell

    def _transition(
        self, new_mode: HVACMode, sensors: SensorReadings, sim_t: float
    ) -> HVACMode:
        """Execute a state transition and log it."""
        if new_mode == self.mode:
            return self.mode

        self.history.append({
            "sim_time": sim_t,
            "from":     self.mode.name,
            "to":       new_mode.name,
            "T_zone":   sensors.T_zone,
            "CO2":      sensors.CO2_ppm,
            "RH":       sensors.RH_zone,
        })
        self._prev_mode = self.mode
        self.mode = new_mode
        self._mode_entry_time = time.monotonic()
        self._mode_duration_s = 0.0
        return self.mode

    # ─────────────────────────────────────────────────────────────────────
    # Control outputs per mode
    # ─────────────────────────────────────────────────────────────────────

    def get_commands(self, mode: HVACMode = None) -> dict:
        """
        Return a dict of high-level commands based on current (or given) mode.
        These are dispatched to individual PID loops.
        """
        m = mode or self.mode
        cmds = {
            "hp_mode":          "OFF",
            "vent_rate_frac":   0.10,   # minimum ventilation (10%)
            "humidifier":       0.0,
            "dehumidifier":     0.0,
            "recirculation":    1.0,
        }

        if m == HVACMode.HEATING:
            cmds["hp_mode"]       = "HEATING"
            cmds["vent_rate_frac"] = 0.15

        elif m == HVACMode.COOLING:
            cmds["hp_mode"]        = "COOLING"
            cmds["vent_rate_frac"] = 0.15

        elif m == HVACMode.VENTILATE:
            cmds["hp_mode"]        = "OFF"
            cmds["vent_rate_frac"] = 1.0   # full ventilation

        elif m == HVACMode.DEHUMIDIFY:
            cmds["hp_mode"]       = "COOLING"   # cooling also dehumidifies
            cmds["dehumidifier"]  = 1.0

        elif m == HVACMode.HUMIDIFY:
            cmds["humidifier"]    = 1.0

        elif m == HVACMode.DEFROST:
            cmds["hp_mode"]       = "HEATING"
            cmds["vent_rate_frac"] = 0.05   # minimal ventilation

        elif m == HVACMode.EMERGENCY:
            cmds["hp_mode"]        = "OFF"
            cmds["vent_rate_frac"] = 0.0
            cmds["recirculation"]  = 0.5

        elif m == HVACMode.STANDBY:
            pass   # defaults already set (minimal vent + everything off)

        return cmds

    def __repr__(self):
        return (
            f"HVACStateMachine(mode={self.mode.name}, "
            f"dwell={self._mode_duration_s:.0f}s)"
        )
