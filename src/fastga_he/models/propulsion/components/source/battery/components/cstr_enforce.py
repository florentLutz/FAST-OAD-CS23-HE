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
class ConstraintsSOCEnforce(om.Group):
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

        self.add_subsystem(
            name="capacity_constraint",
            subsys=ConstraintsSOCCapacity(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="capacity_c_rate",
            subsys=ConstraintsSOCCRate(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="picker",
            subsys=ConstraintsSOCPicker(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )


class ConstraintsSOCCapacity(om.ExplicitComponent):
    """
    Class that enforces one of the constraints on number of module : that the minimum SOC seen by
    the battery during the mission is used for the sizing, ensuring a fitted design of each
    component.
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

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier",
            val=1.0,
            desc="Multiplier for the the number of module if the capacity of the battery is the "
            "active constraint",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        soc_min_mission = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min"
        ]
        soc_min_required = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC"
        ]

        multiplicative_factor = 1.0 + (soc_min_required - soc_min_mission) / 100.0

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier"
        ] = multiplicative_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
        ] = (
            -1.0 / 100.0
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
        ] = (
            1.0 / 100.0
        )


class ConstraintsSOCCRate(om.ExplicitComponent):
    """
    Class that enforces one of the constraints on number of module : that the maximum C-rate seen by
    the battery during the mission is used for the sizing, ensuring a fitted design of each
    component.
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

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier",
            val=1.0,
            desc="Multiplier for the the number of module if the c-rate of the battery is the "
            "active constraint",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        max_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max"
        ]
        c_rate_caliber = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + "cell:c_rate_caliber"
        ]

        multiplicative_factor = max_c_rate / c_rate_caliber

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier"
        ] = multiplicative_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        max_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max"
        ]
        c_rate_caliber = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + "cell:c_rate_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
        ] = (
            1.0 / c_rate_caliber
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + "cell:c_rate_caliber",
        ] = (
            -max_c_rate / c_rate_caliber ** 2.0
        )


class ConstraintsSOCPicker(om.ExplicitComponent):
    """
    Class that enforces the most constraining of the constraints on number of module.
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
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier",
            val=1.0,
            desc="Multiplier for the the number of module if the c-rate of the battery is the "
            "active constraint",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier",
            val=1.0,
            desc="Multiplier for the the number of module if the capacity of the battery is the "
            "active constraint",
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
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":number_modules_limiter",
            val=2.0 / 3.0,
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
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":c_rate_multiplier",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":capacity_multiplier",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
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

        reduction_limiter = inputs[
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":number_modules_limiter"
        ]

        number_of_module_before = number_cells_tot / number_cells_module

        multiplicative_factor_capa = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier"
        ]
        multiplicative_factor_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier"
        ]
        multiplicative_factor = max(multiplicative_factor_capa, multiplicative_factor_c_rate)

        # In order to avoid reducing too much the number of cells at one time, (increasing it
        # should not be an issue), we cap this factor at 2/3 (arbitrary)
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

        reduction_limiter = inputs[
            "convergence:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":number_modules_limiter"
        ]

        number_of_module_before = number_cells_tot / number_cells_module

        multiplicative_factor_capa = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier"
        ]
        multiplicative_factor_c_rate = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier"
        ]
        multiplicative_factor = max(multiplicative_factor_capa, multiplicative_factor_c_rate)

        # In order to avoid reducing too much the number of cells at one time, (increasing it
        # should not be an issue), we cap this factor at 2/3 (arbitrary)
        multiplicative_factor = np.clip(multiplicative_factor, reduction_limiter, None)

        if multiplicative_factor == reduction_limiter:
            partials_capa = 1e-6
            partials_c_rate = 1e-6
        elif multiplicative_factor_capa > multiplicative_factor_c_rate:
            partials_capa = number_cells_tot / number_cells_module
            partials_c_rate = 1e-6
        else:
            partials_capa = 1e-6
            partials_c_rate = number_cells_tot / number_cells_module

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = (
            -number_cells_tot / number_cells_module ** 2.0 * multiplicative_factor
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_cells",
        ] = (
            multiplicative_factor / number_cells_module
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_multiplier",
        ] = partials_c_rate
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":capacity_multiplier",
        ] = partials_capa
