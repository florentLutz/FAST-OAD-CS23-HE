# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCADCDCConverterProdWeightPerFU


class PreLCADCDCConverter(om.Group):
    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCADCDCConverterProdWeightPerFU(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
