# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_DC_LINE_CURRENT, SUBMODEL_CONSTRAINTS_DC_LINE_VOLTAGE


class ConstraintsHarness(om.Group):
    """
    Class that gather the different constraints for the harness be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        option_harness_id = {"harness_id": self.options["harness_id"]}

        self.add_subsystem(
            name="constraints_current_harness",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_LINE_CURRENT, options=option_harness_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_harness",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_LINE_VOLTAGE, options=option_harness_id
            ),
            promotes=["*"],
        )
