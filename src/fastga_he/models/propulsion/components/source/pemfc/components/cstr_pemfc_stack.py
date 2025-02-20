# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA


class ConstraintsPEMFCStack(om.Group):
    """
    Class that gather the different constraints for the PEMFC, ensure or enforce depends on user's
    choice.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
        )

    def setup(self):
        options_constraints = {
            "pemfc_stack_id": self.options["pemfc_stack_id"],
            "max_current_density": self.options["max_current_density"],
        }

        self.add_subsystem(
            name="constraints_pemfc_effective_area",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA, options=options_constraints
            ),
            promotes=["*"],
        )
