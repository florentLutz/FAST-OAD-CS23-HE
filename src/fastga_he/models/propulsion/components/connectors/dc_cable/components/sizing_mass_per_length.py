# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMassPerLength(om.ExplicitComponent):
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
            + ":properties:density",
            val=8960.0,
            units="kg/m**3",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:section",
            units="m**2",
            val=np.nan,
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
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:density",
            units="kg/m**3",
            val=950.0,
            desc="High density polyethylene for cable sheath",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:mass_per_length",
            units="kg/m",
            val=2000.0,
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

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]
        rho_in = inputs["settings:propulsion:he_power_train:DC_cable_harness:insulation:density"]
        rho_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ]
        rho_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:density"]

        m_cable = rho_c * cs_c + rho_in * cs_in + rho_shield * cs_shield + rho_sheath * cs_sheath

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:mass_per_length"
        ] = m_cable

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:mass_per_length"
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
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density",
        ] = cs_c
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:insulation:density"
        ] = cs_in
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ] = cs_shield
        partials[
            output_str, "settings:propulsion:he_power_train:DC_cable_harness:sheath:density"
        ] = cs_sheath

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
        ] = rho_c
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section",
        ] = rho_in
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section",
        ] = rho_shield
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section",
        ] = rho_sheath
