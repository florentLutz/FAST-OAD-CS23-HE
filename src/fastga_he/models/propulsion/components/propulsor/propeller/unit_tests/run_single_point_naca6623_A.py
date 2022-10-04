# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import warnings

from tests.testing_utilities import run_system
from fastga_he.models.propulsion.components.propulsor.propeller.components.performance_point import (
    ComputePropellerPointPerformance,
)

warnings.filterwarnings(action="ignore")

if __name__ == "__main__":

    rpm = 700.0
    j_list = np.array(
        [
            1.196863740662988,
            1.301875701801494,
            1.374490064931894,
            1.442976126543963,
            1.507374896255431,
            1.571718986476590,
            1.632016794414881,
            1.692273592735439,
            1.756535663721135,
            1.812664160523361,
            1.876885221891325,
            1.937018991358687,
            1.993106478543182,
            2.053212908265391,
            2.109259385832154,
            2.169352145681785,
        ]
    )
    ct_list = np.array(
        [
            0.16,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.08,
            0.07,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
        ]
    )
    for j, ct_target in zip(j_list, ct_list):

        ct = ct_target / 2.0
        speed = j * rpm / 60 * 3.048
        iter_count = 0
        twist_75 = 40

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
                "data:geometry:propeller:twist_tip",
                val=22.64504769,
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
                    sections_profile_position_list=[
                        0.0,
                        0.30,
                        0.325,
                        0.35,
                        0.375,
                        0.40,
                        0.425,
                        0.45,
                        0.50,
                    ],
                    sections_profile_name_list=[
                        "naca4436",
                        "naca4430",
                        "naca4424",
                        "naca4420",
                        "naca4418",
                        "naca4415",
                        "naca4414",
                        "naca4412",
                        "naca4409",
                    ],
                ),
                ivc,
            )
            thrust = problem.get_val(
                "data:aerodynamics:propeller:point_performance:thrust", units="N"
            )
            power = problem.get_val(
                "data:aerodynamics:propeller:point_performance:power", units="W"
            )
            efficiency = problem.get_val("data:aerodynamics:propeller:point_performance:efficiency")

            ct = thrust / 1.225 / (rpm / 60) ** 2.0 / 3.048 ** 4
            cp = power / 1.225 / (rpm / 60) ** 3.0 / 3.048 ** 5

            twist_75 *= 1.0 - 2.0 * ct_target * (ct - ct_target) / ct_target
            iter_count += 1

        print(ct, cp, twist_75)
