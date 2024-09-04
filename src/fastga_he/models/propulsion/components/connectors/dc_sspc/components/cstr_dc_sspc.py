# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC, SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC


class ConstraintsDCSSPC(om.Group):
    """
    Class that gather the different constraints for the DC SSPC be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        option_dc_sspc_id = {"dc_sspc_id": self.options["dc_sspc_id"]}

        self.add_subsystem(
            name="constraints_current_dc_sspc",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC, options=option_dc_sspc_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_dc_sspc",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC, options=option_dc_sspc_id
            ),
            promotes=["*"],
        )
