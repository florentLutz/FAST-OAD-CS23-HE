# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingCoolantTotalWeight(om.ExplicitComponent):
    """
    Computation of the coolant tank weight including all the coolant in the TMS.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_tank_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":tank_mass",
            units="kg",
            val=np.nan,
        )
        self.add_input(
            "coolant_density",
            val=np.nan,
            units="kg/m**3",
            desc="Density of the coolant",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":mass",
            units="kg",
            val=3.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]

        coolant_volume = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume"
        ]
        tank_mass = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":tank_mass"
        ]
        coolant_density = inputs["coolant_density"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":mass"
        ] = tank_mass + coolant_volume * coolant_density

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]

        coolant_volume = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume"
        ]
        coolant_density = inputs["coolant_density"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume",
        ] = coolant_density

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":tank_mass",
        ] = 1.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":mass",
            "coolant_density",
        ] = coolant_volume
