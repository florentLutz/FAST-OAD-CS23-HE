# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO


import openmdao.api as om
import fastoad.api as oad
import importlib.util

from fastga.models.aerodynamics.components.elevator_aero import ComputeDeltaElevator
from fastga.models.aerodynamics.components.fuselage.compute_cm_alpha_fus import (
    ComputeCmAlphaFuselage,
)
from .correct_rta_variable_naming import CorrectRTANaming
from .compute_simple_rta_variables import ComputeRTAVariable
from .rta_propulsion_weight import RTAPropulsionWeight
from .rta_aero_approximation import AeroApproximation


@oad.RegisterOpenMDAOSystem("fastga_he.rta_variables")
class RTAVariables(om.Group):
    """
    Gather all the bridging components for FAST-GA-HE and FAST-OAD-RTA.
    """

    def initialize(self):
        if importlib.util.find_spec("rta") is None:
            raise ImportError("RTA needs to be installed before using RTA translation components")

        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )

    def setup(self):
        self.add_subsystem(name="correct_naming", subsys=CorrectRTANaming(), promotes=["*"])
        self.add_subsystem(name="compute_variable", subsys=ComputeRTAVariable(), promotes=["*"])
        self.add_subsystem(
            name="propulsion_weight",
            subsys=RTAPropulsionWeight(power_train_file_path=self.options["power_train_file_path"]),
            promotes=["*"],
        )
        self.add_subsystem(name="correct_elevator", subsys=ComputeDeltaElevator(), promotes=["*"])
        self.add_subsystem(
            name="correct_fus_cm_alpha", subsys=ComputeCmAlphaFuselage(), promotes=["*"]
        )
        self.add_subsystem(name="aero_approx", subsys=AeroApproximation(), promotes=["data:*"])
