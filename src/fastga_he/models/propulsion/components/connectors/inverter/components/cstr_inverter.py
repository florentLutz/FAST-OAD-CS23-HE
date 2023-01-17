# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_INVERTER_CURRENT,
    SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE,
    SUBMODEL_CONSTRAINTS_INVERTER_LOSSES,
    SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY,
)

import openmdao.api as om
import fastoad.api as oad


class ConstraintsInverter(om.Group):
    """
    Class that regroups all of the sub components for the computation of the inverter constraints.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        option_inverter_id = {"inverter_id": inverter_id}

        self.add_subsystem(
            name="constraints_current_inverter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_INVERTER_CURRENT, options=option_inverter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage_inverter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE, options=option_inverter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_losses_inverter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_INVERTER_LOSSES, options=option_inverter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_frequency_inverter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY, options=option_inverter_id
            ),
            promotes=["*"],
        )
