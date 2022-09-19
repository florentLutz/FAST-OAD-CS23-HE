# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from tests.testing_utilities import run_system
from fastga_he.models.propulsion.components.propulsor.propeller.components.performance_point import (
    ComputePropellerPointPerformance,
)

if __name__ == "__main__":

    j = 2.153125
    rpm = 700.0

    speed = j * rpm / 60 * 3.048

    ct_target = 0.01
    ct = ct_target / 2.0
    twist_75 = 40
    iter_count = 0

    while abs(ct - ct_target) / ct_target > 1e-4 and iter_count < 50:

        ivc = om.IndepVarComp()
        ivc.add_output("data:geometry:propeller:diameter", val=3.048, units="m")
        ivc.add_output("data:geometry:propeller:hub_diameter", val=0.6096, units="m")
        ivc.add_output("data:geometry:propeller:blades_number", val=3)
        ivc.add_output("data:geometry:propeller:average_rpm", val=rpm, units="rpm")

        ivc.add_output(
            "data:aerodynamics:propeller:point_performance:twist_75", val=twist_75, units="deg"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:point_performance:twist_75_ref", val=5, units="deg"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:point_performance:rho", val=1.225, units="kg/m**3"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:point_performance:speed", val=speed, units="m/s"
        )
        radius_ratio = np.array(
            [0.0, 0.197, 0.297, 0.400, 0.584, 0.597, 0.708, 0.803, 0.900, 0.949, 0.99]
        )
        ivc.add_output(
            "data:geometry:propeller:radius_ratio_vect",
            val=radius_ratio,
        )
        ivc.add_output(
            "data:geometry:propeller:twist_vect_ref",
            val=np.array(
                [
                    34.01214457,
                    34.01214457,
                    31.33590222,
                    27.3504266,
                    19.627439,
                    19.13878771,
                    16.02529171,
                    14.02445889,
                    12.36807196,
                    11.38711638,
                    11.36709688,
                ]
            ),
            units="deg",
        )

        ivc.add_output(
            "data:geometry:propeller:chord_vect",
            val=[
                0.1149,
                0.1149,
                0.1607,
                0.2094,
                0.2262,
                0.2238,
                0.1984,
                0.1713,
                0.1366,
                0.1162,
                0.1158,
            ],
            units="m",
        )
        ivc.add_output(
            "data:geometry:propeller:sweep_vect",
            val=np.full_like(radius_ratio, 0.0),
            units="rad",
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            ComputePropellerPointPerformance(
                elements_number=100,
                sections_profile_position_list=[0.0, 0.30, 0.35, 0.40, 0.45, 0.50],
                sections_profile_name_list=[
                    "naca4436",
                    "naca4424",
                    "naca4418",
                    "naca4414",
                    "naca4412",
                    "naca4409",
                ],
            ),
            ivc,
        )
        thrust = problem.get_val("data:aerodynamics:propeller:point_performance:thrust", units="N")
        power = problem.get_val("data:aerodynamics:propeller:point_performance:power", units="W")
        efficiency = problem.get_val("data:aerodynamics:propeller:point_performance:efficiency")

        ct = thrust / 1.225 / (rpm / 60) ** 2.0 / 3.048 ** 4
        cp = power / 1.225 / (rpm / 60) ** 3.0 / 3.048 ** 5

        print(ct, twist_75)

        twist_75 *= 1.0 - 2.0 * ct_target * (ct - ct_target) / ct_target
        iter_count += 1

    print(ct, cp, twist_75)
