# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad


from ..constants import SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION
from .sizing_pemfc_volume import SizingPEMFCVolume
from .sizing_pemfc_cg_x import SizingPEMFCCGX
from .sizing_pemfc_cg_y import SizingPEMFCCGY
from .sizing_pemfc_drag import SizingPEMFCDrag
from .cstr_pemfc_stack import ConstraintsPEMFCStack

from ..constants import POSSIBLE_POSITION


class SizingPEMFCStack(om.Group):
    """Class that regroups all the subcomponents for the sizing of the PEMFC stack."""

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="underbelly",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pemfc, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        option_weight_dimension = {
            "pemfc_stack_id": pemfc_stack_id,
        }

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_pemfc",
            subsys=ConstraintsPEMFCStack(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_weight_dimension",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION, options=option_weight_dimension
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_volume",
            subsys=SizingPEMFCVolume(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_CG_x",
            subsys=SizingPEMFCCGX(pemfc_stack_id=pemfc_stack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_CG_y",
            subsys=SizingPEMFCCGY(pemfc_stack_id=pemfc_stack_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "pemfc_drag_ls" if low_speed_aero else "pemfc_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPEMFCDrag(
                    pemfc_stack_id=pemfc_stack_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
