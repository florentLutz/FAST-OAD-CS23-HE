# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE,
    SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN,
    SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY,
    SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES,
)


class ConstraintsRectifier(om.Group):
    """
    Class that gather the different constraints for the rectifier be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            types=str,
            allow_none=False,
        )

    def setup(self):

        option_rectifier_id = {"rectifier_id": self.options["rectifier_id"]}

        self.add_subsystem(
            name="constraint_ac_current_rms_1_phase",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE, options=option_rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraint_ac_voltage_peak",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN, options=option_rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraint_frequency",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY, options=option_rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraint_losses",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES, options=option_rectifier_id
            ),
            promotes=["*"],
        )
