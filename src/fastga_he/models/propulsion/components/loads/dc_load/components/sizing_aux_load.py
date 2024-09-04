# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import openmdao.api as om

from .cstr_aux_load import ConstraintsDCAuxLoad

from ..components.sizing_aux_load_weight import SizingDCAuxLoadWeight
from ..components.sizing_aux_load_cg_y import SizingDCAuxLoadCGY
from ..components.sizing_aux_load_cg_x import SizingDCAuxLoadCGX
from ..components.sizing_aux_load_drag import SizingDCAuxLoadDrag

from ..constants import POSSIBLE_POSITION


class SizingDCAuxLoad(om.Group):
    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_front",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the auxiliary load, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        position = self.options["position"]
        aux_load_id = self.options["aux_load_id"]

        self.add_subsystem(
            name="constraints",
            subsys=ConstraintsDCAuxLoad(aux_load_id=aux_load_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "weight",
            SizingDCAuxLoadWeight(aux_load_id=aux_load_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "aux_load_cg_x",
            SizingDCAuxLoadCGX(aux_load_id=aux_load_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "aux_load_cg_y",
            SizingDCAuxLoadCGY(aux_load_id=aux_load_id, position=position),
            promotes=["data:*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "aux_load_drag_ls" if low_speed_aero else "aux_load_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDCAuxLoadDrag(
                    aux_load_id=aux_load_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
