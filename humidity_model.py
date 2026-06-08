"""
humidity_model.py  &  air_quality_model.py
==========================================
Moisture mass balance and indoor air quality (CO₂, VOC) models.

Humidity Model:
  Zone moisture balance (humidity ratio ω [kg_w / kg_dry_air]):
    M_air * dω/dt = ṁ_vent*(ω_supply - ω_zone)
                  + ṁ_infil*(ω_outdoor - ω_zone)
                  + G_occ   [occupant moisture generation]
                  + G_humidifier - G_dehumidifier

Air Quality Model (CO₂):
  Zone CO₂ balance [ppm]:
    V * dC/dt = Q_vent*(C_supply - C_zone)
              + G_occ * n_occ   [occupant CO₂ generation]
              + G_infiltration

Author: HVAC Control Project
"""

import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
# PSYCHROMETRICS (standalone utilities)
# ═══════════════════════════════════════════════════════════════════════════

def saturation_pressure(T_celsius: float) -> float:
    """
    Antoine equation for water vapour saturation pressure [Pa].
    Valid -40°C to 80°C.
    """
    T = T_celsius
    # Magnus formula
    return 611.2 * np.exp(17.67 * T / (T + 243.5))


def humidity_ratio_from_RH(T_celsius: float, RH: float, P_atm: float = 101325) -> float:
    """
    Humidity ratio ω [kg_w/kg_da] from dry-bulb temp and relative humidity.
    RH in fraction (0–1).
    """
    p_sat = saturation_pressure(T_celsius)
    p_w   = RH * p_sat
    return 0.62198 * p_w / (P_atm - p_w)


def RH_from_humidity_ratio(T_celsius: float, omega: float, P_atm: float = 101325) -> float:
    """
    Relative humidity (0–1) from dry-bulb temp and humidity ratio.
    """
    p_w   = omega * P_atm / (0.62198 + omega)
    p_sat = saturation_pressure(T_celsius)
    return np.clip(p_w / p_sat, 0.0, 1.0)


def dew_point(T_celsius: float, RH: float) -> float:
    """Magnus approximation for dew point [°C]. RH in fraction."""
    if RH <= 0:
        return -99.0
    a, b = 17.67, 243.5
    alpha = np.log(max(RH, 1e-6)) + a * T_celsius / (b + T_celsius)
    return b * alpha / (a - alpha)


# ═══════════════════════════════════════════════════════════════════════════
# HUMIDITY MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HumidityModelParams:
    # Occupant moisture generation (ASHRAE 62.1)
    G_occ_kg_per_hr:   float = 0.05    # kg/hr per person (typical office/home)
    # Max humidifier / dehumidifier capacity
    cap_humidifier_kg_hr:   float = 2.0   # kg/hr (steam or ultrasonic)
    cap_dehumidifier_kg_hr: float = 3.0   # kg/hr (refrigerant or desiccant)


class HumidityModel:
    """
    Zone moisture mass balance.

    State: omega_zone [kg_w / kg_da]
    """

    def __init__(
        self,
        volume_m3: float,
        params: HumidityModelParams = None,
        omega_init: float = 0.008,  # ~50% RH at 20°C
    ):
        self.volume = volume_m3
        self.p = params or HumidityModelParams()
        # Air mass [kg_da]
        self.M_air = volume_m3 * 1.20  # ρ_air ≈ 1.20 kg/m³
        self.omega = omega_init
        self.history = {"omega": [], "RH": []}

    def step(
        self,
        dt: float,
        T_zone: float,
        omega_outdoor: float,
        omega_supply: float,
        mdot_ventilation: float,   # kg_da/s
        mdot_infiltration: float,  # kg_da/s
        n_occupants: float,
        humidifier_cmd: float,     # 0–1 fraction of capacity
        dehumidifier_cmd: float,   # 0–1 fraction of capacity
    ) -> tuple[float, float]:
        """
        Advance humidity one timestep.

        Returns (omega_zone, RH_zone).
        """
        G_occ = n_occupants * self.p.G_occ_kg_per_hr / 3600.0  # kg/s

        G_hum  = humidifier_cmd   * self.p.cap_humidifier_kg_hr   / 3600.0
        G_dehum = dehumidifier_cmd * self.p.cap_dehumidifier_kg_hr / 3600.0

        # Moisture flows [kg_w/s]
        dw_vent  = mdot_ventilation  * (omega_supply  - self.omega)
        dw_infil = mdot_infiltration * (omega_outdoor - self.omega)

        domega_dt = (dw_vent + dw_infil + G_occ + G_hum - G_dehum) / self.M_air

        self.omega = max(0.0, self.omega + domega_dt * dt)
        RH = RH_from_humidity_ratio(T_zone, self.omega)

        self.history["omega"].append(self.omega)
        self.history["RH"].append(RH)

        return self.omega, RH

    @property
    def RH(self, T_zone: float = 22.0) -> float:
        return RH_from_humidity_ratio(T_zone, self.omega)


# ═══════════════════════════════════════════════════════════════════════════
# AIR QUALITY MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AirQualityParams:
    # CO₂ generation per person (ASHRAE 62.1, office activity)
    CO2_per_person_L_min:  float = 0.30    # L/min
    # CO₂ concentration of outdoor air
    CO2_outdoor_ppm:       float = 415.0   # ppm (current atmospheric)
    # CO₂ of supply air (after ERV — between outdoor and exhaust)
    CO2_supply_ppm:        float = 415.0   # ppm (fresh air = outdoor)
    # VOC generation per person
    VOC_per_person_ug_s:   float = 20.0    # µg/s
    # Background VOC (materials off-gassing)
    VOC_background_ug_s:   float = 5.0     # µg/s per 100 m² floor area
    # HEPA filter removal efficiency (if air recirculation active)
    HEPA_efficiency:       float = 0.997


class AirQualityModel:
    """
    CO₂ and VOC concentration model for a zone.

    States:
      C_CO2  [ppm]
      C_VOC  [µg/m³]
    """

    def __init__(
        self,
        volume_m3: float,
        floor_area_m2: float,
        params: AirQualityParams = None,
        C_CO2_init: float = 500.0,
        C_VOC_init:  float = 50.0,
    ):
        self.volume = volume_m3
        self.floor_area = floor_area_m2
        self.p = params or AirQualityParams()
        self.C_CO2 = C_CO2_init   # ppm
        self.C_VOC = C_VOC_init   # µg/m³
        self.history = {"CO2": [], "VOC": []}

    def step(
        self,
        dt: float,
        n_occupants: float,
        Q_ventilation_m3_s: float,   # fresh air volumetric flow [m³/s]
        Q_recirculation_m3_s: float, # recirculated + filtered air [m³/s]
        C_CO2_supply: float = None,  # if None, use outdoor ppm
    ) -> tuple[float, float]:
        """
        Advance CO₂ and VOC one timestep.

        Returns (C_CO2_ppm, C_VOC_ug_m3).
        """
        if C_CO2_supply is None:
            C_CO2_supply = self.p.CO2_supply_ppm

        # Ventilation flow rates [m³/s → air changes / s]
        ach_vent = Q_ventilation_m3_s / self.volume
        ach_recirc = Q_recirculation_m3_s / self.volume

        # ── CO₂ ──
        G_CO2_occ = n_occupants * self.p.CO2_per_person_L_min / 60 * 1e6 / self.volume
        # ppm/s from occupants = (L/s) * 1e6 / V_m3
        # convert L/s → m³/s : /1000; then ppm = m³_CO2/m³_air * 1e6
        G_CO2_occ_ppm_s = (
            n_occupants * self.p.CO2_per_person_L_min / 60.0 / 1000.0
            * 1e6 / self.volume
        )

        dCO2_dt = (
            G_CO2_occ_ppm_s
            + ach_vent * (C_CO2_supply - self.C_CO2)
        )

        # ── VOC ──
        G_VOC_occ  = n_occupants * self.p.VOC_per_person_ug_s / self.volume
        G_VOC_mat  = self.p.VOC_background_ug_s * self.floor_area / 100.0 / self.volume

        # HEPA filtration removes VOC via recirculated air
        removal_recirc = ach_recirc * self.p.HEPA_efficiency * self.C_VOC

        dVOC_dt = (
            G_VOC_occ + G_VOC_mat
            - ach_vent * self.C_VOC        # dilution by ventilation
            - removal_recirc               # HEPA removal
        )

        self.C_CO2 = max(self.p.CO2_outdoor_ppm, self.C_CO2 + dCO2_dt * dt)
        self.C_VOC = max(0.0, self.C_VOC + dVOC_dt * dt)

        self.history["CO2"].append(self.C_CO2)
        self.history["VOC"].append(self.C_VOC)

        return self.C_CO2, self.C_VOC

    @property
    def IAQ_index(self) -> float:
        """
        Composite IAQ index (0=excellent, 100=hazardous).
        Based on ASHRAE 62.1 / WELL Standard thresholds.
        """
        co2_score = np.clip((self.C_CO2 - 400) / 600, 0, 1) * 50
        voc_score = np.clip(self.C_VOC / 500, 0, 1) * 50
        return co2_score + voc_score
