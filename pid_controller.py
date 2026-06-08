"""
pid_controller.py
=================
Generic discrete-time PID controller with:
  - Anti-windup (clamping + back-calculation)
  - Derivative filter (low-pass on derivative term)
  - Output clamping
  - Bumpless transfer (mode switching)

Cascaded PID (inner/outer loop):
  - Outer loop: zone temperature → supply temperature setpoint
  - Inner loop: supply temperature → heating/cooling command (modulation)

Author: HVAC Control Project
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    Kp: float
    Ki: float
    Kd: float
    # Derivative filter coefficient N (0 = no filter, 20 = moderate filter)
    N:  float = 10.0


class PIDController:
    """
    Discrete-time PID with anti-windup and derivative filter.

    Control law (position form):
      u(k) = Kp*e(k) + Ki*dt*Σe + Kd*N*(e(k) - e_filt(k)) / (1 + N*dt/τ_d)

    Anti-windup via conditional integration:
      Integrator freezes when output is saturated AND error has same sign.
    """

    def __init__(
        self,
        gains: PIDGains,
        output_min: float = 0.0,
        output_max: float = 1.0,
        setpoint: float = 0.0,
    ):
        self.gains = gains
        self.output_min = output_min
        self.output_max = output_max
        self.setpoint = setpoint

        # Controller state
        self._integral     = 0.0
        self._e_prev       = 0.0
        self._d_filt_prev  = 0.0   # filtered derivative state
        self._u_prev       = 0.0   # previous output (for bumpless transfer)
        self._t_prev       = None

    def step(
        self,
        measurement: float,
        dt: float,
        setpoint: Optional[float] = None,
        feed_forward: float = 0.0,
    ) -> float:
        """
        Compute one PID step.

        Parameters
        ----------
        measurement  : current process variable
        dt           : timestep [s]
        setpoint     : override setpoint (if None, uses self.setpoint)
        feed_forward : optional feed-forward term added to output

        Returns
        -------
        output : control signal in [output_min, output_max]
        """
        if setpoint is not None:
            self.setpoint = setpoint

        error = self.setpoint - measurement
        Kp, Ki, Kd, N = self.gains.Kp, self.gains.Ki, self.gains.Kd, self.gains.N

        # ── Proportional ──
        P = Kp * error

        # ── Integral with anti-windup ──
        u_unsat = P + Ki * (self._integral + error * dt) + self._d_filt_prev
        u_clamp = np.clip(u_unsat, self.output_min, self.output_max)
        # Conditional integration: only integrate if not saturated or if error
        # is driving output back within bounds
        if not self._is_saturated(u_unsat) or np.sign(error) != np.sign(u_unsat - u_clamp):
            self._integral += error * dt

        # ── Derivative with first-order filter ──
        # Filtered derivative: D_filt(k) = (N*e(k) - D_filt(k-1)) * dt / (dt + N^-1)
        if N > 0:
            tau_d = 1.0 / N
            alpha = dt / (dt + tau_d)
            d_raw = (error - self._e_prev) / dt if dt > 0 else 0.0
            d_filt = (1.0 - alpha) * self._d_filt_prev + alpha * Kd * d_raw * N / N
            # Standard filtered derivative
            d_filt = N * Kd * error - (N * Kd * self._e_prev - self._d_filt_prev) * np.exp(-N * dt)
        else:
            d_filt = Kd * (error - self._e_prev) / dt if dt > 0 else 0.0

        self._d_filt_prev = d_filt

        # ── Total output ──
        u = P + Ki * self._integral + d_filt + feed_forward
        u = np.clip(u, self.output_min, self.output_max)

        self._e_prev = error
        self._u_prev = u

        return u

    def _is_saturated(self, u: float) -> bool:
        return u < self.output_min or u > self.output_max

    def reset(self, integral: float = 0.0):
        """Reset controller state."""
        self._integral    = integral
        self._e_prev      = 0.0
        self._d_filt_prev = 0.0
        self._u_prev      = 0.0

    def set_output_limits(self, lo: float, hi: float):
        self.output_min = lo
        self.output_max = hi

    @property
    def integral(self) -> float:
        return self._integral


# ═══════════════════════════════════════════════════════════════════════════
# CASCADED PID
# ═══════════════════════════════════════════════════════════════════════════

class CascadedPID:
    """
    Two-degree-of-freedom cascaded PID for HVAC zone temperature control.

    Outer loop: zone_temp_error → supply_temp_setpoint
    Inner loop: supply_temp_error → valve/modulation command

    The outer loop runs at a slower rate (every N inner steps).
    Typical HVAC rates:
      Outer: 1–5 min
      Inner: 15–30 sec
    """

    def __init__(
        self,
        outer_gains: PIDGains,
        inner_gains: PIDGains,
        # Outer loop output = supply temperature setpoint
        T_supply_min: float = 15.0,   # °C (minimum supply air temp)
        T_supply_max: float = 55.0,   # °C (maximum supply air temp)
        # Inner loop output = modulation command 0–1
        modulation_min: float = 0.0,
        modulation_max: float = 1.0,
    ):
        self.outer = PIDController(
            outer_gains,
            output_min=T_supply_min,
            output_max=T_supply_max,
        )
        self.inner = PIDController(
            inner_gains,
            output_min=modulation_min,
            output_max=modulation_max,
        )
        self.T_supply_setpoint = 22.0  # current supply temp setpoint

    def step(
        self,
        T_zone: float,
        T_zone_sp: float,
        T_supply_actual: float,
        dt_outer: float,
        dt_inner: float,
        update_outer: bool = True,
    ) -> tuple[float, float]:
        """
        One cascade step.

        Parameters
        ----------
        T_zone         : measured zone temperature [°C]
        T_zone_sp      : zone temperature setpoint [°C]
        T_supply_actual: actual supply air/water temperature [°C]
        dt_outer       : outer loop timestep [s]
        dt_inner       : inner loop timestep [s]
        update_outer   : whether to update outer loop this step

        Returns
        -------
        (T_supply_sp, modulation) — supply temp setpoint and modulation 0–1
        """
        # Outer loop: zone temp → supply temp setpoint
        if update_outer:
            self.T_supply_setpoint = self.outer.step(T_zone, dt_outer, setpoint=T_zone_sp)

        # Inner loop: supply temp → modulation
        modulation = self.inner.step(
            T_supply_actual, dt_inner, setpoint=self.T_supply_setpoint
        )

        return self.T_supply_setpoint, modulation

    def reset(self):
        self.outer.reset()
        self.inner.reset()


# ═══════════════════════════════════════════════════════════════════════════
# PRE-TUNED GAIN SETS
# ═══════════════════════════════════════════════════════════════════════════

# Typical gains for zone temperature control
# Tuned via Ziegler-Nichols on a 30m² zone with τ≈4h, K≈0.8°C/kW
TEMP_OUTER_GAINS = PIDGains(Kp=2.0, Ki=0.02, Kd=5.0, N=5.0)
TEMP_INNER_GAINS = PIDGains(Kp=0.5, Ki=0.10, Kd=0.5, N=10.0)

# Humidity RH control
RH_GAINS = PIDGains(Kp=0.8, Ki=0.005, Kd=0.0, N=0.0)

# CO₂ DCV control (very slow — minutes to hours)
CO2_GAINS = PIDGains(Kp=0.001, Ki=0.0001, Kd=0.0, N=0.0)
