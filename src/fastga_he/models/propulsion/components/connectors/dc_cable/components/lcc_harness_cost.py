# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCHarnessCost(om.ExplicitComponent):
    """
    Computation of cable purchase cost based on the raw material price and gross margin of the
    cable manufacturing industry.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.price_metal = None

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            units="m",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
            units="m",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:density",
            val=8960.0,
            units="kg/m**3",
        )

        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
            units="m",
            val=0.2e-3,
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            units="m",
            val=0.2e-2,
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            units="m",
            val=np.nan,
            desc="Length of the harness",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_unit",
            units="USD",
            val=200.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        r_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ]
        t_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness"
        ]
        t_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness"
        ]
        t_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness"
        ]

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]
        length = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]

        cost_ins = 40.0  # USD/kg

        if rho_c < 8960.0:
            self.price_metal = 2.5  # USD/kg
        else:
            self.price_metal = 10.33  # USD/kg

        c_conductor = np.pi * self.price_metal * (r_c**2.0)
        c_i = np.pi * cost_ins * ((2.0 * r_c + t_in) * t_in)
        c_shield = np.pi * self.price_metal * ((2.0 * (r_c + t_in) + t_shield) * t_shield)
        c_sheath = np.pi * cost_ins * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath)

        c_cable = (c_conductor + c_i + c_shield + c_sheath) * length

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_unit"
        ] = c_cable / 0.55

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_unit"
        )

        r_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ]
        t_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness"
        ]
        t_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness"
        ]
        t_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness"
        ]

        length = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]

        cost_ins = 40.0  # USD/kg

        c_conductor = np.pi * self.price_metal * (r_c**2.0)
        c_i = np.pi * cost_ins * ((2.0 * r_c + t_in) * t_in)
        c_shield = np.pi * self.price_metal * ((2.0 * (r_c + t_in) + t_shield) * t_shield)
        c_sheath = np.pi * cost_ins * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath)

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (c_conductor + c_i + c_shield + c_sheath) / 0.55

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = (
            2.0
            * np.pi
            * length
            * (
                self.price_metal * r_c
                + cost_ins * t_in
                + self.price_metal * t_shield
                + cost_ins * t_sheath
            )
        ) / 0.55
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = (
            2.0
            * np.pi
            * length
            * (cost_ins * (t_in + r_c) + self.price_metal * t_shield + cost_ins * t_sheath)
            / 0.55
        )
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = (
            2.0
            * np.pi
            * length
            * (self.price_metal * (t_shield + r_c + t_in) + cost_ins * t_sheath)
            / 0.55
        )
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
        ] = 2.0 * np.pi * cost_ins * length * (r_c + t_in + t_shield + t_sheath) / 0.55
