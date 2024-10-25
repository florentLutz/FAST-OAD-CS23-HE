# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCATurboshaftProdWeightPerFU
from .pre_lca_use_emission_per_fu import PreLCATurboshaftUseEmissionPerFU, SPECIES_LIST


class PreLCATurboshaft(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.species_list = SPECIES_LIST

    def initialize(self):
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        turboshaft_id = self.options["turboshaft_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCATurboshaftProdWeightPerFU(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emission_per_fu",
            subsys=PreLCATurboshaftUseEmissionPerFU(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )
