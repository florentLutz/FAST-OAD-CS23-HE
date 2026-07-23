# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAPEMFCStackProdWeightPerFU
from .pre_lca_use_emission_per_fu import PreLCAPEMFCStackUseEmissionPerFU, SPECIES_LIST


class PreLCAPEMFCStack(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.species_list = SPECIES_LIST

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAPEMFCStackProdWeightPerFU(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emission_per_fu",
            subsys=PreLCAPEMFCStackUseEmissionPerFU(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
