# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_BATTERY_SOC

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_BATTERY_SOC,
    "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.enforce",
)
class ConstraintsSOCEnforce(om.ExplicitComponent):
    """
    Class that enforces that the minimum SOC seen by the battery during the mission is used for the
    sizing, ensuring a fitted design of each component. For now only enforces that the number of
    module is readjusted to match the minimum SOC level required.
    """

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
            val=np.nan,
            units="percent",
            desc="Minimum state-of-charge of the battery during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
            val=np.nan,
            units="percent",
            desc="Minimum state-of-charge that the battery can have without degradation",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            val=np.nan,
            desc="Number of cells in series inside one battery module",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
            val=np.nan,
            desc="Total number of cells in the battery pack",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=200.0,
            desc="Number of modules in parallel inside the battery pack",
            lower=5.0,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        number_cells_tot = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells"
        ]
        number_cells_module = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells"
        ]
        soc_min_mission = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min"
        ]
        soc_min_required = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC"
        ]

        number_of_module_before = number_cells_tot / number_cells_module

        multiplicative_factor = 1.0 + (soc_min_required - soc_min_mission) / 100.0

        # In order to avoid reducing too much the number of cells at one time, (increasing it
        # should not be an issue), we cap this factor at 2/3 (arbitrary)
        multiplicative_factor = np.clip(multiplicative_factor, 2.0 / 3.0, None)

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
        ] = (number_of_module_before * multiplicative_factor)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        number_cells_tot = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells"
        ]
        number_cells_module = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells"
        ]
        soc_min_mission = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min"
        ]
        soc_min_required = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC"
        ]

        number_of_module_before = number_cells_tot / number_cells_module
        multiplicative_factor = 1.0 + (soc_min_required - soc_min_mission) / 100.0

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
        ] = np.where(
            multiplicative_factor < 2.0 / 3.0,
            1e-6,
            (1.0 + (soc_min_required - soc_min_mission) / 100.0) / number_cells_module,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = np.where(
            multiplicative_factor < 2.0 / 3.0,
            1e-6,
            -number_cells_tot
            * (1.0 + (soc_min_required - soc_min_mission) / 100.0)
            / number_cells_module ** 2.0,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
        ] = np.where(multiplicative_factor < 2.0 / 3.0, 1e-6, -number_of_module_before / 100)
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
        ] = np.where(multiplicative_factor < 2.0 / 3.0, 1e-6, number_of_module_before / 100)
