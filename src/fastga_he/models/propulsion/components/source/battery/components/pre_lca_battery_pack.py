# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from .pre_lca_use_emission_per_fu import PreLCABatteryUseEmissionPerFU, SPECIES_LIST

from ..constants import SERVICE_BATTERY_LIFESPAN, SERVICE_BATTERY_MASS_PER_FU


class PreLCABatteryPack(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.species_list = SPECIES_LIST

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        options_dict = {"battery_pack_id": battery_pack_id}

        if oad.RegisterSubmodel.active_models[SERVICE_BATTERY_LIFESPAN]:
            self.add_subsystem(
                name="battery_life_cycle",
                subsys=oad.RegisterSubmodel.get_submodel(
                    SERVICE_BATTERY_LIFESPAN, options=options_dict
                ),
                promotes=["data:*"],
            )

        self.add_subsystem(
            name="weight_per_fu",
            subsys=oad.RegisterSubmodel.get_submodel(
                SERVICE_BATTERY_MASS_PER_FU, options=options_dict
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_per_fu",
            subsys=PreLCABatteryUseEmissionPerFU(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
