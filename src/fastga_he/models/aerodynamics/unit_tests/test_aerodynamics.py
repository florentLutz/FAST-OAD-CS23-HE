#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import pytest

import numpy as np
import openmdao.api as om

from ..components.cd0_wing_rta import Cd0Wing
from ..components.cd0_fuselage_rta import Cd0Fuselage

from tests.testing_utilities import run_system


def test_cd0_wing_rta():
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:MAC:length", val=2.3, units="m")
    ivc.add_output("data:geometry:wing:area", val=57.2, units="m**2")
    ivc.add_output("data:geometry:wing:wetted_area", val=102.0, units="m**2")
    ivc.add_output("data:geometry:wing:sweep_25", val=0.04, units="rad")
    ivc.add_output("data:geometry:wing:thickness_ratio", val=0.137, units="unitless")
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", val=6480282, units="unitless")
    ivc.add_output(
        "data:aerodynamics:aircraft:cruise:CL", val=np.linspace(0, 1.49, 150), units="unitless"
    )
    ivc.add_output("data:TLAR:cruise_mach", val=0.5, units="unitless")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Cd0Wing(), ivc)

    assert problem.get_val("data:aerodynamics:wing:cruise:CD0") == pytest.approx(
        0.01013465, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_cd0_fuselage_rta():
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:fuselage:length", val=22.39, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_height", val=2.93, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_width", val=2.79, units="m")
    ivc.add_output("data:geometry:fuselage:wetted_area", val=166.018, units="m**2")
    ivc.add_output("data:geometry:wing:area", val=57.2, units="m**2")
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", val=6480282, units="unitless")
    ivc.add_output(
        "data:aerodynamics:aircraft:cruise:CL", val=np.linspace(0, 1.49, 150), units="unitless"
    )
    ivc.add_output("data:TLAR:cruise_mach", val=0.5, units="unitless")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Cd0Fuselage(), ivc)

    assert problem.get_val("data:aerodynamics:fuselage:cruise:CD0") == pytest.approx(
        0.00650617, rel=1e-2
    )

    problem.check_partials(compact_print=True)
