# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAHighRPMICEProdWeightPerFU
from .pre_lca_use_emission_per_fu import PreLCAHighRPMICEUseEmissionPerFU, SPECIES_LIST


class PreLCAHighRPMICE(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.species_list = SPECIES_LIST

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAHighRPMICEProdWeightPerFU(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_per_fu",
            subsys=PreLCAHighRPMICEUseEmissionPerFU(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
