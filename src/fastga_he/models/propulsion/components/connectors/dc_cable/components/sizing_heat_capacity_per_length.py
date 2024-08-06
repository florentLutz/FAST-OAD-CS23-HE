# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
            + ":conductor:radius",
            units="m",
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
            + ":insulation:thickness",
            val=np.nan,
            units="m",
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
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
            units="m",
            val=0.2e-3,
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
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            units="m",
            val=0.2e-2,
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

        hc_conductor = np.pi * cp_c * rho_c * (r_c**2.0)
        hc_i = np.pi * cp_in * rho_in * ((2.0 * r_c + t_in) * t_in)
        hc_shield = np.pi * cp_shield * rho_shield * ((2.0 * (r_c + t_in) + t_shield) * t_shield)
        hc_sheath = (
            np.pi * cp_sheath * rho_sheath * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath)
        )

        hc_cable = hc_conductor + hc_i + hc_shield + hc_sheath

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length"
        ] = hc_cable

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:heat_capacity_per_length"
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
        ] = np.pi * (r_c**2.0) * rho_c
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:specific_heat",
        ] = np.pi * ((2.0 * r_c + t_in) * t_in) * rho_in
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:specific_heat",
        ] = np.pi * ((2.0 * (r_c + t_in) + t_shield) * t_shield) * rho_shield
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:specific_heat",
        ] = np.pi * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath) * rho_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density",
        ] = np.pi * (r_c**2.0) * cp_c
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:density",
        ] = np.pi * ((2.0 * r_c + t_in) * t_in) * cp_in
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density",
        ] = np.pi * ((2.0 * (r_c + t_in) + t_shield) * t_shield) * cp_shield
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:density",
        ] = np.pi * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath) * cp_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = (
            2.0
            * np.pi
            * (
                cp_c * rho_c * r_c
                + cp_in * rho_in * t_in
                + cp_shield * rho_shield * t_shield
                + cp_sheath * t_sheath * rho_sheath
            )
        )
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = (
            2.0
            * np.pi
            * (
                cp_in * rho_in * (t_in + r_c)
                + cp_shield * rho_shield * t_shield
                + cp_sheath * rho_sheath * t_sheath
            )
        )
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = (
            2.0
            * np.pi
            * (cp_shield * rho_shield * (t_shield + r_c + t_in) + cp_sheath * rho_sheath * t_sheath)
        )
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
        ] = 2.0 * np.pi * cp_sheath * rho_sheath * (r_c + t_in + t_shield + t_sheath)
