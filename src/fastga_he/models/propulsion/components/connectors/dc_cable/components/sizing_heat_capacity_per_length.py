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
            + ":conductor:section",
            units="m**2",
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
            + ":insulation:section",
            units="m**2",
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
            + ":shield:section",
            units="m**2",
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
            + ":sheath:section",
            units="m**2",
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

        cs_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        cs_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section"
        ]
        cs_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section"
        ]
        cs_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section"
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

        hc_conductor = cp_c * rho_c * cs_c
        hc_i = cp_in * rho_in * cs_in
        hc_shield = cp_shield * rho_shield * cs_shield
        hc_sheath = cp_sheath * rho_sheath * cs_sheath

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

        cs_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        cs_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section"
        ]
        cs_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section"
        ]
        cs_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section"
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
        ] = rho_c * cs_c
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:specific_heat",
        ] = rho_in * cs_in
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat",
        ] = rho_shield * cs_shield
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:sheath:specific_heat"
        ] = rho_sheath * cs_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density",
        ] = cp_c * cs_c
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:insulation:density"
        ] = cp_in * cs_in
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ] = cp_shield * cs_shield
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:sheath:density"
        ] = cp_sheath * cs_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
        ] = rho_c * cp_c
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section",
        ] = rho_in * cp_in
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section",
        ] = rho_shield * cp_shield
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section",
        ] = rho_sheath * cp_sheath
