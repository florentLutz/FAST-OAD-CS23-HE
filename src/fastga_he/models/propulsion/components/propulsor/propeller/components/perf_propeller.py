# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_blade_speed import PerformancesBladesSpeeds
from .perf_blade_aoa import PerformancesBladesAngleOfAttack
from .perf_blade_aero import PerformancesBladesAero
from .perf_bemt import PerformancesBEMT
from .perf_tip_loss import PerformancesTipLoss
from .perf_adt import PerformancesADT, PerformancesADTToLoop
from .perf_induced_speed_convergence import PerformancesInducedSpeedConvergence


class PerformancesPropeller(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("elements_number", default=7, types=int)
        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        self.add_subsystem(
            name="blade_speed",
            subsys=PerformancesBladesSpeeds(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="blade_aoa",
            subsys=PerformancesBladesAngleOfAttack(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="blade_aero",
            subsys=PerformancesBladesAero(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="blade_bemt",
            subsys=PerformancesBEMT(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tip_loss",
            subsys=PerformancesTipLoss(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
        # self.add_subsystem(
        #     name="perf_adt",
        #     subsys=PerformancesADT(
        #         number_of_points=number_of_points,
        #         elements_number=elements_number,
        #         propeller_id=propeller_id,
        #     ),
        #     promotes=["*"],
        # )
        # self.add_subsystem(
        #     name="speed_convergence",
        #     subsys=PerformancesInducedSpeedConvergence(
        #         number_of_points=number_of_points,
        #         elements_number=elements_number,
        #     ),
        #     promotes=["*"],
        # )
        self.add_subsystem(
            name="perf_adt",
            subsys=PerformancesADTToLoop(
                number_of_points=number_of_points,
                elements_number=elements_number,
                propeller_id=propeller_id,
            ),
            promotes=["*"],
        )
