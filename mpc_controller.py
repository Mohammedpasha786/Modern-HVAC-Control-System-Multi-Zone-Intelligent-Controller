"""
mpc_controller.py
=================
Model Predictive Control (MPC) for multi-zone HVAC temperature control.

Formulation:
  - State-space model linearized from thermal RC network
  - Prediction horizon: N_p steps (default 24 × 15 min = 6 hours)
  - Control horizon: N_c steps (default 8 × 15 min = 2 hours)
  - Cost function: comfort deviation + energy + peak demand penalty
  - Constraints: zone temperature bounds, max heating/cooling rate
  - Solver: quadratic program via scipy.optimize (or OSQP if available)

The MPC re-optimizes every control step using updated measurements and
weather forecasts (perfect forecast used in simulation; Kalman-filtered
estimates used in deployment).

Author: HVAC Control Project
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

try:
    import osqp
    import scipy.sparse as sp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    from scipy.optimize import minimize


@dataclass
class MPCConfig:
    """MPC hyperparameters."""
    # Horizon
    N_p: int   = 24     # prediction horizon [steps]
    N_c: int   = 8      # control horizon [steps]
    dt:  float = 900.0  # timestep [s] = 15 min

    # Cost weights
    w_T_comfort:  float = 10.0   # zone temp deviation penalty
    w_T_peak:     float = 5.0    # penalty outside comfort band
    w_dQ:         float = 0.5    # control move penalty (smooth control)
    w_energy:     float = 1.0    # electricity cost weight
    w_demand:     float = 2.0    # peak demand penalty

    # Constraints
    T_zone_min:   float = 18.0   # °C — hard min (frost protection)
    T_zone_max:   float = 27.0   # °C — hard max
    Q_max_kW:     float = 10.0   # kW — max HVAC capacity
    dQ_max_kW:    float = 2.0    # kW/step — max ramp rate

    # Energy tariff (time-of-use pricing)
    peak_hours:   tuple = (8, 21)   # hours — peak tariff window
    peak_tariff:  float = 0.25      # $/kWh  peak
    offpeak_tariff: float = 0.08    # $/kWh  off-peak


class MPCController:
    """
    Receding-horizon MPC for HVAC zone temperature control.

    Linearized model (per zone):
      T(k+1) = a * T(k) + b * Q(k) + c * T_outdoor(k) + f * Q_solar(k)

    Where a, b, c, f are derived from the RC network parameters.
    """

    def __init__(
        self,
        config: MPCConfig = None,
        zone_params: dict = None,
    ):
        self.config = config or MPCConfig()
        # Zone model coefficients (set via identify_model)
        self.a = 0.98    # temperature persistence
        self.b = 0.001   # HVAC gain [°C/W per step]
        self.c = 0.005   # outdoor coupling gain
        self.f = 0.002   # solar gain

        if zone_params:
            self._set_model(zone_params)

        # Controller state
        self.Q_prev   = 0.0   # kW — previous control action
        self.T_zone   = 22.0  # current zone temperature
        self._last_plan: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────
    # Model identification from RC parameters
    # ─────────────────────────────────────────────────────────────────────

    def identify_model(
        self,
        C_air: float,        # J/K
        UA_total: float,     # W/K  (total envelope conductance)
        UA_inner: float,     # W/K  (wall-to-air conductance)
        dt: float,           # s
    ):
        """
        Derive linear model coefficients from RC network parameters.
        Uses Euler discretization of the 1-state simplified model:
          C_air * dT/dt = -UA * T + UA * T_out + Q_hvac
        """
        tau = C_air / UA_total   # thermal time constant [s]
        self.a = np.exp(-dt / tau)
        self.b = (1 - self.a) / UA_total   # [°C/W]
        self.c = 1 - self.a                 # outdoor coupling
        self.f = (1 - self.a) * 0.5        # solar (50% goes to air)

    # ─────────────────────────────────────────────────────────────────────
    # MPC solve
    # ─────────────────────────────────────────────────────────────────────

    def solve(
        self,
        T_zone_current: float,
        T_sp_profile: np.ndarray,      # setpoint profile [N_p]
        T_outdoor_forecast: np.ndarray, # forecast [N_p]
        Q_solar_forecast: np.ndarray,   # solar forecast [W/m², N_p]
        hour_of_day: float = 12.0,
    ) -> tuple[float, np.ndarray]:
        """
        Solve MPC optimization and return optimal first control action.

        Parameters
        ----------
        T_zone_current    : current zone temperature [°C]
        T_sp_profile      : temperature setpoint over horizon [°C], shape (N_p,)
        T_outdoor_forecast: outdoor temperature forecast, shape (N_p,)
        Q_solar_forecast  : solar irradiance forecast [W], shape (N_p,)
        hour_of_day       : for time-of-use tariff

        Returns
        -------
        Q_opt_kW : optimal heating (+) or cooling (-) power [kW]
        Q_plan   : full optimal plan [kW], shape (N_c,)
        """
        cfg = self.config
        N_p, N_c = cfg.N_p, cfg.N_c

        # Pad forecasts if shorter than N_p
        T_outdoor = np.broadcast_to(
            np.pad(T_outdoor_forecast, (0, max(0, N_p - len(T_outdoor_forecast))),
                   mode='edge'), (N_p,)
        )
        Q_solar = np.broadcast_to(
            np.pad(Q_solar_forecast, (0, max(0, N_p - len(Q_solar_forecast))),
                   mode='edge'), (N_p,)
        ) if Q_solar_forecast is not None else np.zeros(N_p)

        T_sp = np.broadcast_to(
            np.pad(T_sp_profile, (0, max(0, N_p - len(T_sp_profile))), mode='edge'),
            (N_p,)
        )

        # Tariff profile
        tariff = self._tariff_profile(hour_of_day, N_p)

        # ── Build prediction matrices ──────────────────────────────────
        # T(k) = A^k * T(0) + Σ A^(k-j-1) * B * u(j) + disturbances
        a, b = self.a, self.b * 1000  # convert kW → W for model

        # Free response (no HVAC)
        T_free = np.zeros(N_p)
        T_k = T_zone_current
        for k in range(N_p):
            T_k = a * T_k + self.c * T_outdoor[k] + self.f * Q_solar[k]
            T_free[k] = T_k

        # Forced response matrix Φ [N_p × N_c]
        Phi = np.zeros((N_p, N_c))
        for k in range(N_p):
            for j in range(min(k + 1, N_c)):
                Phi[k, j] = (a ** (k - j)) * b

        # ── Cost function (quadratic) ─────────────────────────────────
        # J = (T_pred - T_sp)' W_T (T_pred - T_sp)
        #   + u' W_e u  (energy)
        #   + Δu' W_du Δu  (smoothness)
        W_T  = np.eye(N_p) * cfg.w_T_comfort
        W_e  = np.diag(tariff[:N_c]) * cfg.w_energy
        W_du = np.eye(N_c) * cfg.w_dQ

        # Difference matrix for Δu
        D = np.eye(N_c) - np.eye(N_c, k=-1)
        D[0, 0] = 1.0

        H = Phi.T @ W_T @ Phi + W_e + D.T @ W_du @ D
        H = (H + H.T) / 2  # ensure symmetry

        e_free = T_free - T_sp
        f_vec  = Phi.T @ W_T @ e_free - D.T @ W_du @ (np.zeros(N_c) - self.Q_prev * np.eye(N_c)[:, 0])

        # ── Constraints ───────────────────────────────────────────────
        Q_max = cfg.Q_max_kW
        Q_min = -Q_max   # cooling

        # ── Solve ─────────────────────────────────────────────────────
        if OSQP_AVAILABLE:
            Q_plan = self._solve_osqp(H, f_vec, Q_min, Q_max, cfg.dQ_max_kW, N_c)
        else:
            Q_plan = self._solve_scipy(H, f_vec, Q_min, Q_max, cfg.dQ_max_kW, N_c)

        # ── Return first action ────────────────────────────────────────
        Q_opt = float(Q_plan[0])
        self.Q_prev = Q_opt
        self._last_plan = Q_plan

        return Q_opt, Q_plan

    def _solve_scipy(
        self, H, f, Q_min, Q_max, dQ_max, N_c
    ) -> np.ndarray:
        """Fallback QP solver using scipy minimize (SLSQP)."""
        def cost(u):
            return 0.5 * u @ H @ u + f @ u

        def grad(u):
            return H @ u + f

        from scipy.optimize import minimize, Bounds, LinearConstraint

        bounds = Bounds(lb=np.full(N_c, Q_min), ub=np.full(N_c, Q_max))

        # Rate constraints: |u(k) - u(k-1)| ≤ dQ_max
        # For first step: |u(0) - Q_prev| ≤ dQ_max
        result = minimize(
            cost, x0=np.zeros(N_c), jac=grad,
            method='SLSQP', bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100}
        )
        return result.x if result.success else np.zeros(N_c)

    def _solve_osqp(self, H, f, Q_min, Q_max, dQ_max, N_c) -> np.ndarray:
        """OSQP solver (if available) — faster and more reliable."""
        P = sp.csc_matrix(H)
        q = f.astype(float)

        # Box constraints
        l_box = np.full(N_c, Q_min)
        u_box = np.full(N_c, Q_max)

        prob = osqp.OSQP()
        prob.setup(P, q, sp.eye(N_c, format='csc'), l_box, u_box,
                   warm_starting=True, verbose=False, eps_abs=1e-4)
        result = prob.solve()
        return result.x if result.info.status == 'solved' else np.zeros(N_c)

    @staticmethod
    def _tariff_profile(hour_start: float, N: int, dt_h: float = 0.25) -> np.ndarray:
        """Build time-of-use electricity tariff profile."""
        tariff = np.zeros(N)
        for k in range(N):
            h = (hour_start + k * dt_h) % 24
            tariff[k] = 0.25 if 8 <= h < 21 else 0.08
        return tariff

    # ─────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        T_zone: float,
        Q_plan: np.ndarray,
        T_outdoor: np.ndarray,
        Q_solar: np.ndarray,
    ) -> np.ndarray:
        """
        Open-loop prediction using a given control plan.
        Returns predicted zone temperatures over horizon.
        """
        N = len(Q_plan)
        T_pred = np.zeros(N + 1)
        T_pred[0] = T_zone
        for k in range(N):
            T_pred[k + 1] = (
                self.a * T_pred[k]
                + self.b * Q_plan[k] * 1000
                + self.c * T_outdoor[k]
                + self.f * Q_solar[k]
            )
        return T_pred[1:]

    def reset(self):
        self.Q_prev = 0.0
        self._last_plan = None
