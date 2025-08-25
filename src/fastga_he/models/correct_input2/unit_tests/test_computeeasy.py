#doesn't work
import numpy as np
import pytest

import openmdao.api as om

from ..components.compute_easy import EASY_compute

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "problem_outputs_from_RTA_mio2.xml"

def test_diameter():
    ivc = get_indep_var_comp(
        list_inputs(EASY_compute(pmsm_id="motor_1")), __file__, XML_FILE
    )
    ivc.add_output("data:propulsion:he_power_train:ACPMSM:motor_1:Form_coefficient", val=0.6)
    ivc.add_output(
        "data:propulsion:he_power_train:ACPMSM:motor_1:Tangential_stress", val=50000, units="N/m**2"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(EASY_compute(pmsm_id="motor_1"), ivc)