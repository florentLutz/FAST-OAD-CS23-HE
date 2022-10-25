# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import pytest
import plotly.graph_objects as go

from ..components.sizing_energy_coefficient_scaling import SizingInverterLossCoefficientScaling
from ..components.sizing_energy_coefficients import SizingInverterEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingInverterResistanceScaling
from ..components.sizing_reference_resistance import SizingInverterResistances
from ..components.sizing_thermal_resistance import SizingInverterThermalResistances
from ..components.sizing_thermal_resistance_casing import SizingInverterCasingThermalResistance
from ..components.sizing_weight_casing import SizingInverterCasingsWeight
from ..components.sizing_heat_capacity_casing import SizingInverterCasingsWeight
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_resistance import PerformancesResistance
from ..components.perf_conduction_loss import PerformancesConductionLosses
from ..components.perf_total_loss import PerformancesLosses
from ..components.perf_temperature_derivative import PerformancesTemperatureDerivative
from ..components.perf_temperature_increase import PerformancesTemperatureIncrease
from ..components.perf_temperature import PerformancesTemperature
from ..components.perf_inverter import PerformancesInverter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_inverter.xml"
NB_POINTS_TEST = 10


def test_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterLossCoefficientScaling(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingInverterLossCoefficientScaling(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:a"
    ) == pytest.approx(1.385, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:c"
    ) == pytest.approx(0.722, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterEnergyCoefficients(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterEnergyCoefficients(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:a"
    ) == pytest.approx(0.02197006, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:a"
    ) == pytest.approx(0.0058837, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:a"
    ) == pytest.approx(0.02087697, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:b"
    ) == pytest.approx(3.326e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:b"
    ) == pytest.approx(0.000340, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:b"
    ) == pytest.approx(0.000254, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:c"
    ) == pytest.approx(3.707e-7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:c"
    ) == pytest.approx(-3.257e-8, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:c"
    ) == pytest.approx(-1.256e-7, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterResistanceScaling(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterResistanceScaling(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:resistance"
    ) == pytest.approx(1.385, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterResistances(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterResistances(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:resistance", units="ohm"
    ) == pytest.approx(0.00209135, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:resistance", units="ohm"
    ) == pytest.approx(0.00258995, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterThermalResistances(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterThermalResistances(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:thermal_resistance", units="K/W"
    ) == pytest.approx(0.114955, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:thermal_resistance", units="K/W"
    ) == pytest.approx(0.148195, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance_casing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingThermalResistance(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingThermalResistance(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:thermal_resistance", units="K/W"
    ) == pytest.approx(0.021, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_casings():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingsWeight(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingsWeight(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:weight", units="kg"
    ) == pytest.approx(1.0446, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_capacity_casing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingsWeight(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingsWeight(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:heat_capacity", units="J/degK"
    ) == pytest.approx(208.92, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_switching_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSwitchingLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("current", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A")
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesSwitchingLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_losses_igbt = np.array(
        [126.5, 184.4, 250.8, 326.2, 411.0, 505.5, 610.2, 725.5, 851.8, 989.5]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [72.8, 111.0, 156.1, 208.1, 266.8, 332.2, 404.4, 483.1, 568.4, 660.2]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_resistance_profile():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesResistance(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("inverter_temperature", units="degK", val=temperature)

    problem = run_system(
        PerformancesResistance(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_resistance_igbt = (
        np.array([2.05, 2.16, 2.27, 2.36, 2.44, 2.51, 2.56, 2.61, 2.64, 2.66]) * 1e-3
    )
    assert problem.get_val("resistance_igbt", units="ohm") == pytest.approx(
        expected_resistance_igbt, rel=1e-2
    )
    resistance_diode = np.array([2.55, 2.66, 2.77, 2.86, 2.94, 3.01, 3.06, 3.11, 3.14, 3.16]) * 1e-3
    assert problem.get_val("resistance_diode", units="ohm") == pytest.approx(
        resistance_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_conduction_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesConductionLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("current", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A")
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))
    ivc.add_output(
        "resistance_igbt",
        np.array([2.05, 2.16, 2.27, 2.36, 2.44, 2.51, 2.56, 2.61, 2.64, 2.66]) * 1e-3,
        units="ohm",
    )
    ivc.add_output(
        "resistance_diode",
        np.array([2.55, 2.66, 2.77, 2.86, 2.94, 3.01, 3.06, 3.11, 3.14, 3.16]) * 1e-3,
        units="ohm",
    )

    problem = run_system(
        PerformancesConductionLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_losses_igbt = np.array(
        [40.68, 53.82, 69.56, 87.95, 109.3, 133.82, 161.37, 192.63, 227.0, 264.8]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [50.23, 56.79, 62.45, 66.86, 69.83, 71.09, 70.28, 67.34, 61.84, 53.64]
    )
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_inverter():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [126.5, 184.4, 250.8, 326.2, 411.0, 505.5, 610.2, 725.5, 851.8, 989.5],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [72.8, 111.0, 156.1, 208.1, 266.8, 332.2, 404.4, 483.1, 568.4, 660.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [40.68, 53.82, 69.56, 87.95, 109.3, 133.82, 161.37, 192.63, 227.0, 264.8],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [50.23, 56.79, 62.45, 66.86, 69.83, 71.09, 70.28, 67.34, 61.84, 53.64],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1741.26, 2436.06, 3233.46, 4134.66, 5141.58, 6255.66, 7477.5, 8811.42, 10254.24, 11808.84]
    )
    assert problem.get_val("losses_inverter", units="W") == pytest.approx(expected_losses, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_temperature_derivative():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTemperatureDerivative(
                inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    total_losses = np.array(
        [1773.0, 2481.6, 3295.2, 4215.0, 5242.8, 6379.8, 7627.8, 8989.2, 10462.2, 12049.8]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("inverter_temperature", units="degK", val=temperature)
    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(temperature, 288.15))

    problem = run_system(
        PerformancesTemperatureDerivative(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("inverter_temperature_time_derivative", units="degK/s") == pytest.approx(
        np.array([2.83, 0.89, -0.57, -1.54, -2.03, -2.03, -1.54, -0.56, 0.91, 2.88]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_perf_temperature_increase():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    derivative = np.array([2.83, 0.89, -0.57, -1.54, -2.03, -2.03, -1.54, -0.56, 0.91, 2.88])
    ivc.add_output("inverter_temperature_time_derivative", units="degK/s", val=derivative)
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 300.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_increase = np.array(
        [849.0, 267.0, -171.0, -462.0, -609.0, -609.0, -462.0, -168.0, 273.0, 864.0]
    )
    assert (
        problem.get_val(
            "inverter_temperature_increase",
            units="degK",
        )
        == pytest.approx(expected_increase, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_perf_temperature():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTemperature(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    total_losses = np.array(
        [1741.26, 2436.06, 3233.46, 4134.66, 5141.58, 6255.66, 7477.5, 8811.42, 10254.24, 11808.84]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")
    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(total_losses, 288.15))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperature(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature = np.array(
        [300.56, 305.52, 311.22, 317.65, 324.85, 332.81, 341.54, 351.07, 361.39, 372.5]
    )
    assert (
        problem.get_val(
            "inverter_temperature",
            units="degK",
        )
        == pytest.approx(expected_temperature, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_performances_inverter_tot():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output("heat_sink_temperature", units="degK", val=np.full(NB_POINTS_TEST, 288.15))
    ivc.add_output("current", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A")
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )

    expected_losses = np.array(
        [1747.3, 2438.8, 3230.6, 4125.8, 5127.3, 6238.2, 7462.0, 8802.1, 10262.3, 11846.3]
    )
    assert problem.get_val("losses_inverter", units="W") == pytest.approx(expected_losses, rel=1e-2)
    expected_temperature = np.array(
        [300.4, 305.2, 310.8, 317.0, 324.0, 331.8, 340.4, 349.8, 360.0, 371.1]
    )
    assert problem.get_val(
        "component.temperature_inverter.inverter_temperature", units="degK"
    ) == pytest.approx(expected_temperature, rel=1e-2)


def test_map_efficiency():

    nb_points_maps = 50

    current_orig = np.linspace(0.0, 500.0, nb_points_maps)
    switching_frequency_orig = np.linspace(3000.0, 12000.0, nb_points_maps)
    modulation_index_orig = np.linspace(0.0, 1.0, nb_points_maps)
    # current, switching_frequency = np.meshgrid(current_orig, switching_frequency_orig)
    current, modulation_index = np.meshgrid(current_orig, modulation_index_orig)

    current = current.flatten()
    # switching_frequency = switching_frequency.flatten()
    switching_frequency = np.full_like(current, 9000.0)
    temperature = np.full_like(current, 273.5)
    # modulation_index = np.full_like(current, 0.9)
    modulation_index = modulation_index.flatten()

    ivc = get_indep_var_comp(
        list_inputs(PerformancesInverter(inverter_id="inverter_1", number_of_points=current.size)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(current, 288.15))
    ivc.add_output("current", current, units="A")
    ivc.add_output("switching_frequency", switching_frequency)
    ivc.add_output("modulation_index", modulation_index)

    problem = run_system(
        PerformancesInverter(inverter_id="inverter_1", number_of_points=current.size), ivc
    )

    print(problem["losses_inverter"])

    current = current.reshape((nb_points_maps, nb_points_maps))
    switching_frequency = current.reshape((nb_points_maps, nb_points_maps))
    losses = problem["losses_inverter"].reshape((nb_points_maps, nb_points_maps))

    fig = go.Figure()

    losses_contour = go.Contour(
        x=current_orig,
        y=modulation_index_orig,
        z=losses,
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=np.max(losses),
        zmin=np.min(losses),
    )
    fig.add_trace(losses_contour)
    fig.update_layout(
        title_text="Sampled inverter efficiency map",
        title_x=0.5,
        xaxis_title="Current [A]",
        yaxis_title="Modulation index [-]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    # fig.show()
