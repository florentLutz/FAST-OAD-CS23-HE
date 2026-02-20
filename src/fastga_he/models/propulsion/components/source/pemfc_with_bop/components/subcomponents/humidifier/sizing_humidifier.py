# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .sizing_humidifier_volume import SizingHumidifierVolume
from .sizing_humidifier_weight import SizingHumidifierWeight


class SizingHumidifier(om.Group):
    """
    Sizing of the PEMFC Humidifier
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_subsystem(
            "humidifier_volume",
            SizingHumidifierVolume(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "humidifier_weight",
            SizingHumidifierWeight(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
