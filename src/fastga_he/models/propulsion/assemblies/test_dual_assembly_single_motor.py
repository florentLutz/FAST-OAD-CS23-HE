# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import fastoad.api as oad
import numpy as np
import pytest
import copy
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga_he.models.propulsion.assemblers.performances_from_pt_file import (
    PowerTrainPerformancesFromFile,
)
from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

from ..components.connectors.dc_cable.constants import (
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
)
from ..components.connectors.inverter.constants import (
    SUBMODEL_INVERTER_EFFICIENCY,
    SUBMODEL_INVERTER_JUNCTION_TEMPERATURE,
)
from ..components.connectors.dc_dc_converter.constants import SUBMODEL_DC_DC_CONVERTER_EFFICIENCY

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "dual_assembly_single_motor.xml"
NB_POINTS_TEST = 50
COEFF_DIFF = 0.0


@pytest.fixture()
def restore_submodels():
    """
    Since the submodels in the configuration file differ from the defaults, this restore process
    ensures subsequent assembly tests run under default conditions.
    """
    old_submodels = copy.deepcopy(oad.RegisterSubmodel.active_models)
    yield
    oad.RegisterSubmodel.active_models = old_submodels


def test_assembly_from_pt_file(restore_submodels):
    pt_file_path = pth.join(DATA_FOLDER_PATH, "dual_assembly_single_motor.yml")
    network_file_path = pth.join(OUT_FOLDER_PATH, "dual_assembly.html")
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_INVERTER_EFFICIENCY] = (
        "fastga_he.submodel.propulsion.inverter.efficiency.from_losses"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_INVERTER_JUNCTION_TEMPERATURE] = (
        "fastga_he.submodel.propulsion.inverter.junction_temperature.from_losses"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_DC_CONVERTER_EFFICIENCY] = (
        "fastga_he.submodel.propulsion.dc_dc_converter.efficiency.from_losses"
    )
    power_train_network_viewer(pt_file_path, network_file_path)

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=True,
            )
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 50))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    assert problem.get_val("component.propeller_1.shaft_power_in", units="kW") + problem.get_val(
        "component.propeller_2.shaft_power_in", units="kW"
    ) == pytest.approx(
        problem.get_val("component.gearbox_1.shaft_power_in", units="kW") * 0.98, rel=1e-4
    )

    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                92906.3,
                93302.2,
                93699.1,
                94097.2,
                94496.3,
                94896.5,
                95297.9,
                95700.3,
                96103.8,
                96508.3,
                96913.9,
                97320.6,
                97728.3,
                98137.1,
                98546.9,
                98957.8,
                99369.7,
                99782.6,
                100196.5,
                100611.4,
                101027.4,
                101444.3,
                101862.3,
                102281.2,
                102701.1,
                103122.1,
                103543.9,
                103966.8,
                104390.6,
                104815.4,
                105241.2,
                105667.9,
                106095.6,
                106524.2,
                106953.8,
                107384.3,
                107815.7,
                108248.1,
                108681.5,
                109115.8,
                109551.0,
                109987.1,
                110424.2,
                110862.2,
                111301.2,
                111741.1,
                112181.9,
                112623.6,
                113066.3,
                113510.0,
            ]
        ),
        abs=1,
    )
