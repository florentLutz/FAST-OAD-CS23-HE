# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_displacement_volume import SizingICEDisplacementVolume

from ..components.perf_torque import PerformancesTorque
from ..components.perf_equivalent_sl_power import PerformancesEquivalentSeaLevelPower
from ..components.perf_mean_effective_pressure import PerformancesMeanEffectivePressure
from ..components.perf_sfc import PerformancesSFC
from ..components.perf_fuel_consumption import PerformancesICEFuelConsumption
from ..components.perf_fuel_consumed import PerformancesICEFuelConsumed
from ..components.perf_maximum import PerformancesMaximum

from ..components.perf_ice import PerformancesICE

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_ice.xml"
NB_POINTS_TEST = 10


def test_displacement_volume():

    ivc = get_indep_var_comp(
        list_inputs(SizingICEDisplacementVolume(ice_id="ice_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingICEDisplacementVolume(ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:displacement_volume", units="m**3"
    ) == pytest.approx(0.0107, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2700.0, NB_POINTS_TEST),
        units="min**-1",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        np.array([573.0, 610.0, 646.0, 682.0, 717.0, 752.0, 786.0, 819.0, 852.0, 884.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_equivalent_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "altitude",
        val=np.linspace(8000.0, 0.0, NB_POINTS_TEST),
        units="ft",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEquivalentSeaLevelPower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("equivalent_SL_power", units="kW") == pytest.approx(
        np.array([198.0, 206.0, 213.0, 220.0, 226.0, 232.0, 237.0, 242.0, 246.0, 250.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_mean_effective_pressure():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesMeanEffectivePressure(ice_id="ice_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque_out",
        val=np.array([573.0, 610.0, 646.0, 682.0, 717.0, 752.0, 786.0, 819.0, 852.0, 884.0]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMeanEffectivePressure(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("mean_effective_pressure", units="bar") == pytest.approx(
        np.array([13.5, 14.3, 15.2, 16.0, 16.8, 17.7, 18.5, 19.2, 20.0, 20.8]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_sfc():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "mean_effective_pressure",
        val=np.array([13.5, 14.3, 15.2, 16.0, 16.8, 17.7, 18.5, 19.2, 20.0, 20.6]),
        units="bar",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2699.0, NB_POINTS_TEST),
        units="min**-1",
    )
    # If we put 2700.0 the check_partials will clip as intended but cause an unrealistically high
    # value for partial check

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesSFC(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("specific_fuel_consumption", units="g/kW/h") == pytest.approx(
        np.array([246.0, 246.0, 247.0, 249.0, 252.0, 257.0, 262.0, 267.0, 274.0, 280.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumption():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "specific_fuel_consumption",
        val=np.array([246.0, 246.0, 247.0, 249.0, 252.0, 257.0, 262.0, 267.0, 274.0, 280.0]),
        units="g/kW/h",
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumption(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICEFuelConsumed(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "equivalent_SL_power",
        val=np.array([198.0, 206.0, 213.0, 220.0, 226.0, 232.0, 237.0, 242.0, 246.0, 250.0]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesMaximum(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(250e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_ice():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesICE(ice_id="ice_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "shaft_power_out",
        val=np.linspace(150.0, 250.0, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output(
        "rpm",
        val=np.linspace(2500.0, 2699.0, NB_POINTS_TEST),
        units="min**-1",
    )
    ivc.add_output(
        "altitude",
        val=np.linspace(8000.0, 0.0, NB_POINTS_TEST),
        units="ft",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesICE(ice_id="ice_1", number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )
    assert problem.get_val("non_consumable_energy_t", units="kW*h") == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:power_max_SL", units="W"
    ) == pytest.approx(250e3, rel=1e-2)

    problem.check_partials(compact_print=True)
