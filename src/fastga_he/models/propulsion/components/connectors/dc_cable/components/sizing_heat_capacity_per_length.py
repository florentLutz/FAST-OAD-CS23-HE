# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatCapacityPerLength(om.ExplicitComponent):
    """Computation of mass per length of cable."""

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
            + ":conductor:unit_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat",
            val=386.0,
            units="J/kg/degK",
            desc="1.0 for copper, 0.0 for aluminium",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:density",
            val=8960.0,
            units="kg/m**3",
            desc="1.0 for copper, 0.0 for aluminium",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:unit_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:specific_heat",
            units="J/kg/degK",
            val=2800.0,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:density",
            units="kg/m**3",
            val=1450.0,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":shield:unit_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat",
            units="J/kg/degK",
            val=386,
            desc="Copper used for shielding tape",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density",
            units="kg/m**3",
            val=8960,
            desc="High density polyethylene for cable sheath",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":sheath:unit_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:specific_heat",
            units="J/kg/degK",
            val=1550,
            desc="High density polyethylene for cable sheath",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:density",
            units="kg/m**3",
            val=950.0,
            desc="High density polyethylene for cable sheath",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length",
            units="J/degK/m",
            val=7000.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        v_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:unit_volume"
        ]
        v_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:unit_volume"
        ]
        v_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:unit_volume"
        ]
        v_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:unit_volume"
        ]

        cp_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat"
        ]
        cp_in = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation" ":specific_heat"
        ]
        cp_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat"
        ]
        cp_sheath = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:sheath" ":specific_heat"
        ]

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]
        rho_in = inputs["settings:propulsion:he_power_train:DC_cable_harness:insulation:density"]
        rho_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ]
        rho_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:density"]

        hc_conductor = cp_c * rho_c * v_c
        hc_i = cp_in * rho_in * v_in
        hc_shield = cp_shield * rho_shield * v_shield
        hc_sheath = cp_sheath * rho_sheath * v_sheath

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length"
        ] = hc_conductor + hc_i + hc_shield + hc_sheath

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length"
        )

        v_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:unit_volume"
        ]
        v_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:unit_volume"
        ]
        v_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:unit_volume"
        ]
        v_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:unit_volume"
        ]

        cp_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat"
        ]
        cp_in = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation" ":specific_heat"
        ]
        cp_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat"
        ]
        cp_sheath = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:sheath" ":specific_heat"
        ]

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]
        rho_in = inputs["settings:propulsion:he_power_train:DC_cable_harness:insulation:density"]
        rho_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ]
        rho_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:density"]

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat",
        ] = rho_c * v_c
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:specific_heat",
        ] = rho_in * v_in
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat",
        ] = rho_shield * v_shield
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:sheath:specific_heat"
        ] = rho_sheath * v_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density",
        ] = cp_c * v_c
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:insulation:density"
        ] = cp_in * v_in
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ] = cp_shield * v_shield
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:sheath:density"
        ] = cp_sheath * v_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:unit_volume",
        ] = rho_c * cp_c
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:unit_volume",
        ] = rho_in * cp_in
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:unit_volume",
        ] = rho_shield * cp_shield
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:unit_volume",
        ] = rho_sheath * cp_sheath
