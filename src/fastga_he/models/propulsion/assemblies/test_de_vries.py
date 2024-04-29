# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

from utils.filter_residuals import filter_residuals

from ..assemblers.delta_from_pt_file import AerodynamicDeltasFromPTFile

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "verif_de_vries.xml"
NB_POINTS_TEST = 1


def test_slipstream_from_pt_file():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "verif_de_vries.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            AerodynamicDeltasFromPTFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=40.0, units="m/s")
    ivc.add_output("alpha", val=2.0, units="deg")
    ivc.add_output("thrust", val=28.53, units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        AerodynamicDeltasFromPTFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Predicted value in :cite:`de:2019` figure 5)b). There is still quite a bit of difference
    # with the value from the paper, and an even greater difference with the value from
    # experiment (as was the case in the paper). The order of magnitude is however the same so we
    # will keep it like this for now. The difference may stem from the way the paper handle the
    # AOA, we input it, they compute it.

    assert problem.get_val("delta_Cl") == pytest.approx(
        np.array([0.0616]),
        abs=1e-2,
    )
