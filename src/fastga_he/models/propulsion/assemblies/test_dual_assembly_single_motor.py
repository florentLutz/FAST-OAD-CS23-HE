# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pytest
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga_he.models.propulsion.assemblers.performances_from_pt_file import (
    PowerTrainPerformancesFromFile,
)
from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "dual_assembly_single_motor.xml"
NB_POINTS_TEST = 50
COEFF_DIFF = 0.0


def test_assembly_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "dual_assembly_single_motor.yml")
    network_file_path = pth.join(OUT_FOLDER_PATH, "dual_assembly.html")

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
                95158.0,
                95571.4,
                95986.0,
                96401.8,
                96818.7,
                97236.8,
                97655.9,
                98076.2,
                98497.7,
                98920.2,
                99343.8,
                99768.6,
                100194.4,
                100621.3,
                101049.3,
                101478.3,
                101908.4,
                102339.6,
                102771.8,
                103205.0,
                103639.3,
                104074.6,
                104511.0,
                104948.3,
                105386.7,
                105826.1,
                106266.5,
                106707.8,
                107150.2,
                107593.5,
                108037.9,
                108483.2,
                108929.5,
                109376.7,
                109825.0,
                110274.2,
                110724.3,
                111175.4,
                111627.5,
                112080.5,
                112534.5,
                112989.4,
                113445.3,
                113902.1,
                114359.9,
                114818.6,
                115278.3,
                115739.0,
                116200.6,
                116663.2,

            ]
        ),
        abs=1,
    )
