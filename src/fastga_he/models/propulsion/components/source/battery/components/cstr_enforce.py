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
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
            val=np.nan,
            units="h**-1",
            desc="Maximum C-rate of the battery modules during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "cell:c_rate_caliber",
            val=2.4,
            units="h**-1",
            desc="Maximum C-rate that the battery reference cell can provide",
        )

        self.add_input(
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "number_modules_limiter",
            val=1.0 / 3.0,
            desc="Convergence parameter used the reduction of the amount of cell from one loop to the other",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=200.0,
            desc="Number of modules in parallel inside the battery pack",
            lower=5.0,
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + "cell:c_rate_caliber",
            ],
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

        max_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max"
        ]
        c_rate_caliber = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + "cell:c_rate_caliber"
        ]

        reduction_limiter = inputs[
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "number_modules_limiter"
        ]

        number_of_module_before = number_cells_tot / number_cells_module

        multiplicative_factor_capacity = 1.0 + (soc_min_required - soc_min_mission) / 100.0
        multiplicative_factor_c_rate = max_c_rate / c_rate_caliber

        multiplicative_factor = max(multiplicative_factor_capacity, multiplicative_factor_c_rate)

        # In order to avoid reducing too much the number of cells at one time, (increasing it
        # should not be an issue), we cap this factor at 1/3 (arbitrary)
        multiplicative_factor = np.clip(multiplicative_factor, reduction_limiter, None)

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

        max_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max"
        ]
        c_rate_caliber = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + "cell:c_rate_caliber"
        ]

        reduction_limiter = inputs[
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "number_modules_limiter"
        ]

        number_of_module_before = number_cells_tot / number_cells_module
        multiplicative_factor_capacity = 1.0 + (soc_min_required - soc_min_mission) / 100.0
        multiplicative_factor_c_rate = max_c_rate / c_rate_caliber

        multiplicative_factor = max(multiplicative_factor_capacity, multiplicative_factor_c_rate)

        # Depending on which constraints is active, the partials of one will be turned off
        if multiplicative_factor_capacity > multiplicative_factor_c_rate:
            flag_capacity = 1.0
            flag_c_rate = 0.0
        else:
            flag_capacity = 0.0
            flag_c_rate = 1.0

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            reduction_limiter / number_cells_module,
            multiplicative_factor / number_cells_module,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            -number_cells_tot / number_cells_module ** 2.0 * reduction_limiter,
            -number_cells_tot * multiplicative_factor / number_cells_module ** 2.0,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            1e-6,
            -number_of_module_before / 100 * flag_capacity,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            1e-6,
            number_of_module_before / 100 * flag_capacity,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            1e-6,
            number_of_module_before / c_rate_caliber * flag_c_rate,
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "cell:c_rate_caliber",
        ] = np.where(
            multiplicative_factor < reduction_limiter,
            1e-6,
            -number_of_module_before * max_c_rate / c_rate_caliber ** 2.0 * flag_c_rate,
        )
