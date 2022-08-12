# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import pytest
import plotly.graph_objects as go

from ..components.sizing_scaling_ratio import InverterScalingRatio
from ..components.sizing_energy_coefficients import InverterEnergyCoefficients
from ..components.sizing_resistance import InverterResistances
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_conduction_loss import PerformancesConductionLosses
from ..components.perf_total_loss import PerformancesLosses
from ..components.perf_inverter import PerformanceInverter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_inverter.xml"
NB_POINTS_TEST = 10


def test_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(InverterScalingRatio(inverter_id="inverter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(InverterScalingRatio(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:a"
    ) == pytest.approx(1.111, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:c"
    ) == pytest.approx(0.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:resistance"
    ) == pytest.approx(0.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(InverterEnergyCoefficients(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(InverterEnergyCoefficients(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:a"
    ) == pytest.approx(1.5106e-9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:a"
    ) == pytest.approx(-9.204e-10, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:a"
    ) == pytest.approx(-1.07e-10, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:b"
    ) == pytest.approx(3.986e-06, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:b"
    ) == pytest.approx(1.116e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:b"
    ) == pytest.approx(1.33e-05, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:c"
    ) == pytest.approx(0.0010926, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:c"
    ) == pytest.approx(0.0011241, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:c"
    ) == pytest.approx(0.00081576, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(InverterResistances(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(InverterResistances(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:resistance", units="ohm"
    ) == pytest.approx(1.8e-3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:resistance", units="ohm"
    ) == pytest.approx(1.44e-3, rel=1e-2)

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
        [57.3, 103.9, 169.6, 257.6, 371.1, 513.2, 687.0, 895.9, 1142.9, 1431.3]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [33.7, 61.2, 99.9, 151.8, 218.6, 302.3, 404.7, 527.7, 673.2, 843.1]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
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

    problem = run_system(
        PerformancesConductionLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_losses_igbt = np.array(
        [50.6, 65.0, 81.4, 99.9, 120.7, 143.8, 169.4, 197.6, 228.5, 262.2]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array([39.2, 43.4, 46.7, 48.9, 50.1, 50.1, 48.8, 46.1, 42.0, 36.3])
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_inverter():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [57.3, 103.9, 169.6, 257.6, 371.1, 513.2, 687.0, 895.9, 1142.9, 1431.3],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [33.7, 61.2, 99.9, 151.8, 218.6, 302.3, 404.7, 527.7, 673.2, 843.1],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [50.6, 65.0, 81.4, 99.9, 120.7, 143.8, 169.4, 197.6, 228.5, 262.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [39.2, 43.4, 46.7, 48.9, 50.1, 50.1, 48.8, 46.1, 42.0, 36.3],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1084.8, 1641.0, 2385.6, 3349.2, 4563.0, 6056.4, 7859.4, 10003.8, 12519.6, 15437.4]
    )
    assert problem.get_val("losses_inverter", units="W") == pytest.approx(expected_losses, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_inverter_tot():
    ivc = get_indep_var_comp(
        list_inputs(PerformanceInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("current", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A")
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))

    problem = run_system(
        PerformanceInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )

    expected_losses = np.array(
        [1084.8, 1641.0, 2385.6, 3349.2, 4563.0, 6056.4, 7859.4, 10003.8, 12519.6, 15437.4]
    )
    assert problem.get_val("losses_inverter", units="W") == pytest.approx(expected_losses, rel=1e-2)


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
    # modulation_index = np.full_like(current, 0.9)
    modulation_index = modulation_index.flatten()

    ivc = get_indep_var_comp(
        list_inputs(PerformanceInverter(inverter_id="inverter_1", number_of_points=current.size)),
        __file__,
        XML_FILE,
    )

    ivc.add_output("current", current, units="A")
    ivc.add_output("switching_frequency", switching_frequency)
    ivc.add_output("modulation_index", modulation_index)

    problem = run_system(
        PerformanceInverter(inverter_id="inverter_1", number_of_points=current.size), ivc
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
    fig.show()
