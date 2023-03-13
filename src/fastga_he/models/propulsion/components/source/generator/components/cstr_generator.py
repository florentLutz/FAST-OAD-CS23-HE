# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_GENERATOR_TORQUE, SUBMODEL_CONSTRAINTS_GENERATOR_RPM


class ConstraintsGenerator(om.Group):
    """
    Class that gather the different constraints for the generator be they ensure or enforce.
    """

    def initialize(self):

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):

        generator_id = self.options["generator_id"]

        option_generator_id = {"generator_id": generator_id}

        self.add_subsystem(
            name="constraints_torque_generator",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_GENERATOR_TORQUE, options=option_generator_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_rpm_generator",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_GENERATOR_RPM, options=option_generator_id
            ),
            promotes=["*"],
        )
