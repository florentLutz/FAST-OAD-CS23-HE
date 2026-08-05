# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os

import openmdao.api as om

from .components_ducted_fan.perf_rpm import PerformancesRPM
from .components_ducted_fan.perf_power_coefficient import PerformancesPowerCoefficient
from .components_ducted_fan.perf_shaft_power import PerformancesShaftPower
from .components_ducted_fan.perf_torque import PerformancesTorque
from .components_ducted_fan.perf_maximum import PerformancesMaximum

# Both surrogate .pkl files ship inside components_ducted_fan/; default straight to them (via
# __file__, so it works regardless of OS or current working directory) since no YAML "options:"
# block sets surrogate_pkl for ducted_fan_i.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SURROGATE_PKL = os.path.join(_HERE, "components_ducted_fan", "surrogate_smt.pkl")
_DEFAULT_GRAD_SURROGATE_PKL = os.path.join(_HERE, "components_ducted_fan", "surrogate_pt.pkl")


class PerformancesDuctedFan(om.Group):
    """
    Converted from edf_propulsion.py (EDFResidual + EDFOutputs) into the FAST-OAD-CS23-HE
    performance group convention, following ..components.perf_propeller (PerformancesPropeller).

    rpm is now solved implicitly (Newton, handled by the nonlinear_solver set below) instead of
    being read from a mission profile like the propeller's perf_mission_rpm.py -- the ducted fan
    always spins at whatever rpm is needed to meet the required thrust. See perf_rpm.py for
    details and for the approximations carried over from edf_propulsion.py.
    """

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "surrogate_pkl",
            default=_DEFAULT_SURROGATE_PKL,
            desc="Path to the SMT KRG surrogate (.pkl)",
        )
        self.options.declare(
            "grad_surrogate_pkl",
            default=_DEFAULT_GRAD_SURROGATE_PKL,
            desc="Path to the PyTorch MLP surrogate (.pkl)",
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        number_of_points = self.options["number_of_points"]
        surrogate_pkl = self.options["surrogate_pkl"]
        grad_surrogate_pkl = self.options["grad_surrogate_pkl"]

        rpm_group = self.add_subsystem(
            "rpm",
            PerformancesRPM(  # Takes the required thrust as input, solves rpm via Newton
                ducted_fan_id=ducted_fan_id,
                number_of_points=number_of_points,
                surrogate_pkl=surrogate_pkl,
                grad_surrogate_pkl=grad_surrogate_pkl,
            ),
            promotes=["*"],
        )
        rpm_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        rpm_group.nonlinear_solver.options["maxiter"] = 20
        rpm_group.nonlinear_solver.options["atol"] = 1e-2
        rpm_group.nonlinear_solver.options["rtol"] = 1e-8
        rpm_group.nonlinear_solver.options["iprint"] = -1
        rpm_group.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        rpm_group.linear_solver = om.DirectSolver()

        self.add_subsystem(
            "power_coefficient",
            PerformancesPowerCoefficient(
                ducted_fan_id=ducted_fan_id,
                number_of_points=number_of_points,
                surrogate_pkl=surrogate_pkl,
                grad_surrogate_pkl=grad_surrogate_pkl,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "shaft_power",
            PerformancesShaftPower(ducted_fan_id=ducted_fan_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "torque",
            PerformancesTorque(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "maximum",
            PerformancesMaximum(ducted_fan_id=ducted_fan_id, number_of_points=number_of_points),
            promotes=["*"],
        )
