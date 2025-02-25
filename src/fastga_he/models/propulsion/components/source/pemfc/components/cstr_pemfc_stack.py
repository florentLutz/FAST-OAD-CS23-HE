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
            "model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
                 "Aerostak 200W empirical polarization model is set as default.",
        )

    def setup(self):
        options_constraints = {
            "pemfc_stack_id": self.options["pemfc_stack_id"],
            "model_fidelity": self.options["model_fidelity"],
        }

        self.add_subsystem(
            name="constraints_pemfc_effective_area",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA, options=options_constraints
            ),
            promotes=["*"],
        )
