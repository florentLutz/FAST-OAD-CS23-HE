# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.performances.mission.takeoff import TakeOffPhase
from .mission_vector import MissionVector


@oad.RegisterOpenMDAOSystem("fastga_he.performances.mission_vector", domain=ModelDomain.OTHER)
class FullMission(om.Group):
    """Computes and potentially save mission and takeoff based on options."""

    def initialize(self):

        self.options.declare("out_file", default="", types=str)

    def setup(self):

        self.add_subsystem(
            "solve_equilibrium",
            MissionVector(
                out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
