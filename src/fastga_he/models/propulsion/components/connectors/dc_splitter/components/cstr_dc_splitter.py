# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SPLITTER,
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SPLITTER,
)


class ConstraintsDCSplitter(om.Group):
    """
    Class that gather the different constraints for the DC splitter be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            types=str,
            allow_none=False,
        )

    def setup(self):
        option_dc_splitter_id = {"dc_splitter_id": self.options["dc_splitter_id"]}

        self.add_subsystem(
            name="constraints_current_dc_splitter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_CURRENT_DC_SPLITTER, options=option_dc_splitter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_dc_splitter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SPLITTER, options=option_dc_splitter_id
            ),
            promotes=["*"],
        )
