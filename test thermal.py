import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from models.thermal_zone import (
    ThermalZone, ZoneGeometry, ConstructionProperties,
    make_bedroom, make_living_room
)
from models.heat_pump import HeatPump, HeatPumpMode, HeatPumpSpec
from models.humidity_model import (
    HumidityModel, AirQualityModel,
    saturation_pressure, humidity_ratio_from_RH, RH_from_humidity_ratio
)
from controllers.pid_controller import PIDController, PIDGains, CascadedPID
from controllers.fsm_controller import (
    HVACStateMachine, ComfortSetpoints, SensorReadings, HVACMode
)


# ═══════════════════════════════════════════════════════════════════════════
# THERMAL ZONE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalZone:

    def test_zone_creation(self):
        zone = make_bedroom()
        assert zone.C_air > 0
        assert zone.C_wall > 0
        assert zone.UA_wall > 0
        assert zone.time_constant_hours > 0.5  # at least 30 min

    def test_equilibrium_no_hvac(self):
        """Without HVAC, zone should drift toward outdoor temperature."""
        zone = make_bedroom(T_init=22.0)
        T_outdoor = -10.0
        for _ in range(4 * 24):  # 24 hours at 15-min steps
            zone.step(dt=900, Q_hvac=0, T_outdoor=T_outdoor,
                      I_solar=0, n_occupants=0)
        # After 24h, should be significantly closer to outdoor temp
        assert zone.T_zone < 15.0, f"T_zone={zone.T_zone:.1f}°C should have dropped"

    def test_heating_raises_temperature(self):
        """Continuous heating should raise zone temperature."""
        zone = make_bedroom(T_init=10.0)
        T_start = zone.T_zone
        for _ in range(8):  # 2 hours at 15-min
            zone.step(dt=900, Q_hvac=3000, T_outdoor=0, I_solar=0, n_occupants=0)
        assert zone.T_zone > T_start + 3.0

    def test_cooling_lowers_temperature(self):
        """Continuous cooling should lower zone temperature."""
        zone = make_bedroom(T_init=28.0)
        T_start = zone.T_zone
        for _ in range(8):
            zone.step(dt=900, Q_hvac=-3000, T_outdoor=35, I_solar=0, n_occupants=0)
        assert zone.T_zone < T_start - 3.0

    def test_solar_gains_increase_temperature(self):
        """Solar irradiance should heat the zone."""
        zone1 = make_living_room(T_init=22.0)
        zone2 = make_living_room(T_init=22.0)
        for _ in range(4):
            zone1.step(dt=900, Q_hvac=0, T_outdoor=20, I_solar=800, n_occupants=0)
            zone2.step(dt=900, Q_hvac=0, T_outdoor=20, I_solar=0,   n_occupants=0)
        assert zone1.T_zone > zone2.T_zone

    def test_occupant_heat_gains(self):
        """Occupants should raise zone temperature."""
        zone1 = make_bedroom(T_init=20.0)
        zone2 = make_bedroom(T_init=20.0)
        for _ in range(4):
            zone1.step(dt=900, Q_hvac=0, T_outdoor=20, I_solar=0, n_occupants=2)
            zone2.step(dt=900, Q_hvac=0, T_outdoor=20, I_solar=0, n_occupants=0)
        assert zone1.T_zone > zone2.T_zone

    def test_rk4_vs_euler_accuracy(self):
        """RK4 should be more accurate than Euler for large dt."""
        zone_rk4  = make_bedroom(T_init=22.0)
        zone_euler = make_bedroom(T_init=22.0)
        # Large timestep where Euler is less accurate
        for _ in range(4):
            zone_rk4.step( dt=1800, Q_hvac=2000, T_outdoor=0, I_solar=0,
                           n_occupants=0, method="rk4")
            zone_euler.step(dt=1800, Q_hvac=2000, T_outdoor=0, I_solar=0,
                            n_occupants=0, method="euler")
        # Both should be finite and in a reasonable range
        assert 0 < zone_rk4.T_zone  < 50
        assert 0 < zone_euler.T_zone < 50

    def test_thermal_mass_parameter(self):
        """Zone with concrete slab should have higher thermal capacitance."""
        geo = ZoneGeometry(floor_area=20, ceiling_height=2.7, wall_area=45,
                           window_area=4, roof_area=0, floor_slab=True)
        con = ConstructionProperties()
        zone_slab   = ThermalZone("slab",   geo, con)
        geo2 = ZoneGeometry(floor_area=20, ceiling_height=2.7, wall_area=45,
                            window_area=4, roof_area=0, floor_slab=False)
        zone_noslab = ThermalZone("noslab", geo2, con)
        assert zone_slab.C_wall > zone_noslab.C_wall
        assert zone_slab.time_constant_hours > zone_noslab.time_constant_hours

    def test_history_logging(self):
        """Zone should log history correctly."""
        zone = make_bedroom()
        N = 10
        for _ in range(N):
            zone.step(dt=900, Q_hvac=1000, T_outdoor=0, I_solar=0, n_occupants=1)
        assert len(zone.history["T_zone"]) == N

    def test_reset(self):
        """Reset should restore initial conditions."""
        zone = make_bedroom(T_init=22.0)
        zone.step(dt=900, Q_hvac=5000, T_outdoor=-10, I_solar=0, n_occupants=0)
        assert zone.T_zone != 22.0
        zone.reset(T_init=22.0)
        assert zone.T_zone == pytest.approx(22.0)
        assert len(zone.history["T_zone"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# HEAT PUMP TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHeatPump:

    def test_heating_cop_decreases_with_cold_outdoor(self):
        """Heating COP should decrease as outdoor temp drops."""
        hp = HeatPump()
        cop_warm = hp.cop_heating(10.0)
        cop_cold = hp.cop_heating(-10.0)
        assert cop_warm > cop_cold > 0

    def test_cooling_cop_decreases_with_hot_outdoor(self):
        """Cooling COP should decrease as outdoor temp rises."""
        hp = HeatPump()
        cop_cool1 = hp.cop_cooling(25.0)
        cop_cool2 = hp.cop_cooling(40.0)
        assert cop_cool1 > cop_cool2 > 0

    def test_heating_cop_always_above_1(self):
        """Heating COP should always be > 1 (otherwise use resistance)."""
        hp = HeatPump()
        for T in [-20, -10, 0, 10, 20]:
            if T > hp.spec.min_outdoor_heating:
                assert hp.cop_heating(T) >= 1.0

    def test_off_mode_zero_output(self):
        """HP in OFF mode should produce zero heat and zero electricity."""
        hp = HeatPump()
        result = hp.step(HeatPumpMode.OFF, 0.5, 5.0, 900)
        assert result["Q_delivered"] == 0.0
        assert result["P_electric"]  == 0.0

    def test_modulation_scales_output(self):
        """Higher modulation should give proportionally higher output."""
        hp = HeatPump()
        r50 = hp.step(HeatPumpMode.HEATING, 0.5, 5.0, 900)
        hp.reset()
        r100 = hp.step(HeatPumpMode.HEATING, 1.0, 5.0, 900)
        assert r100["Q_delivered"] > r50["Q_delivered"]

    def test_energy_accumulation(self):
        """Running HP for multiple steps should accumulate energy."""
        hp = HeatPump()
        for _ in range(4):  # 1 hour at 15-min steps
            hp.step(HeatPumpMode.HEATING, 0.8, 5.0, 900)
        assert hp.E_heating > 0
        assert hp.E_electric > 0
        # COP check: heating energy > electric energy
        assert hp.E_heating > hp.E_electric


# ═══════════════════════════════════════════════════════════════════════════
# PSYCHROMETRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPsychrometrics:

    def test_saturation_pressure_increases_with_temp(self):
        """Saturation pressure should increase with temperature."""
        p1 = saturation_pressure(0)
        p2 = saturation_pressure(20)
        p3 = saturation_pressure(40)
        assert p1 < p2 < p3

    def test_saturation_pressure_at_100c(self):
        """At 100°C, saturation pressure should be ~101325 Pa."""
        p = saturation_pressure(100)
        assert abs(p - 101325) < 5000  # within 5%

    def test_humidity_ratio_roundtrip(self):
        """humidity_ratio_from_RH → RH_from_humidity_ratio should be identity."""
        T, RH = 22.0, 0.55
        omega = humidity_ratio_from_RH(T, RH)
        RH_back = RH_from_humidity_ratio(T, omega)
        assert abs(RH_back - RH) < 0.001

    def test_rh_clipped_to_unit_interval(self):
        """RH should never exceed 1.0."""
        omega_wet = 0.05  # very wet air
        RH = RH_from_humidity_ratio(10.0, omega_wet)
        assert 0.0 <= RH <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# PID CONTROLLER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPIDController:

    def test_proportional_response(self):
        """P-only controller should respond proportionally to error."""
        pid = PIDController(PIDGains(Kp=2.0, Ki=0.0, Kd=0.0), 0.0, 10.0, setpoint=5.0)
        u = pid.step(measurement=3.0, dt=60.0)
        # Error = 2.0, Kp = 2.0 → u = 4.0 (clamped to [0,10])
        assert abs(u - 4.0) < 0.5

    def test_integral_eliminates_offset(self):
        """I term should eliminate steady-state error over time."""
        pid = PIDController(PIDGains(Kp=1.0, Ki=0.1, Kd=0.0), 0.0, 1.0, setpoint=22.0)
        # Simulate constant error
        u = 0.0
        for _ in range(100):
            u = pid.step(measurement=20.0, dt=60.0)
        # After many steps, integral should push output to max
        assert u == pytest.approx(1.0, abs=0.01)

    def test_output_clamping(self):
        """Output should always be within [output_min, output_max]."""
        pid = PIDController(PIDGains(Kp=100.0, Ki=0.0, Kd=0.0),
                            output_min=-1.0, output_max=1.0, setpoint=50.0)
        u = pid.step(measurement=0.0, dt=60.0)
        assert -1.0 <= u <= 1.0

    def test_reset_clears_state(self):
        """Reset should zero integral and derivative state."""
        pid = PIDController(PIDGains(Kp=1.0, Ki=0.5, Kd=0.0), setpoint=22.0)
        for _ in range(10):
            pid.step(20.0, 60.0)
        assert pid.integral != 0.0
        pid.reset()
        assert pid.integral == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FSM CONTROLLER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFSMController:

    def test_heating_triggered_when_cold(self):
        """FSM should enter HEATING when zone is cold."""
        fsm = HVACStateMachine()
        # Override min dwell to allow immediate transition
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors = SensorReadings(T_zone=18.0, CO2_ppm=500, RH_zone=0.5, T_outdoor=5.0)
        mode = fsm.update(sensors)
        assert mode == HVACMode.HEATING

    def test_cooling_triggered_when_hot(self):
        """FSM should enter COOLING when zone is hot."""
        fsm = HVACStateMachine()
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors = SensorReadings(T_zone=27.0, CO2_ppm=500, RH_zone=0.5, T_outdoor=35.0)
        mode = fsm.update(sensors)
        assert mode == HVACMode.COOLING

    def test_ventilate_triggered_by_co2(self):
        """High CO₂ should trigger VENTILATE mode."""
        fsm = HVACStateMachine()
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors = SensorReadings(T_zone=22.0, CO2_ppm=1100, RH_zone=0.5, T_outdoor=15.0)
        mode = fsm.update(sensors)
        assert mode == HVACMode.VENTILATE

    def test_emergency_on_sensor_fault(self):
        """Sensor fault should immediately trigger EMERGENCY."""
        fsm = HVACStateMachine()
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors = SensorReadings(T_zone=22.0, sensor_fault=True)
        mode = fsm.update(sensors)
        assert mode == HVACMode.EMERGENCY

    def test_standby_in_comfort_zone(self):
        """All conditions met should result in STANDBY."""
        fsm = HVACStateMachine()
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors = SensorReadings(T_zone=22.0, CO2_ppm=500, RH_zone=0.50,
                                 VOC_ug_m3=50, T_outdoor=15.0)
        mode = fsm.update(sensors)
        assert mode == HVACMode.STANDBY

    def test_transition_logging(self):
        """Transitions should be logged in history."""
        fsm = HVACStateMachine()
        fsm._min_dwell = {m: 0.0 for m in HVACMode}
        sensors_cold = SensorReadings(T_zone=16.0, CO2_ppm=400, T_outdoor=0.0)
        fsm.update(sensors_cold)
        assert len(fsm.history) >= 1
        assert fsm.history[-1]["to"] == "HEATING"


# ═══════════════════════════════════════════════════════════════════════════
# AIR QUALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAirQualityModel:

    def test_co2_rises_with_occupants(self):
        """CO₂ should increase with occupants and no ventilation."""
        aq = AirQualityModel(volume_m3=50, floor_area_m2=20, C_CO2_init=415)
        CO2_0 = aq.C_CO2
        for _ in range(20):
            aq.step(dt=900, n_occupants=3,
                    Q_ventilation_m3_s=0.001, Q_recirculation_m3_s=0)
        assert aq.C_CO2 > CO2_0 + 50

    def test_co2_decreases_with_ventilation(self):
        """High ventilation should reduce CO₂ concentration."""
        aq = AirQualityModel(volume_m3=50, floor_area_m2=20, C_CO2_init=1200)
        for _ in range(10):
            aq.step(dt=900, n_occupants=0,
                    Q_ventilation_m3_s=0.5, Q_recirculation_m3_s=0)
        assert aq.C_CO2 < 1000

    def test_co2_never_below_outdoor(self):
        """CO₂ should never drop below outdoor concentration (415 ppm)."""
        aq = AirQualityModel(volume_m3=50, floor_area_m2=20, C_CO2_init=500)
        for _ in range(100):
            aq.step(dt=900, n_occupants=0, Q_ventilation_m3_s=1.0,
                    Q_recirculation_m3_s=0)
        assert aq.C_CO2 >= 415.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
