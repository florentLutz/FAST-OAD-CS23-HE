# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
            + ":conductor:radius",
            units="m",
            val=np.nan,
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
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density",
            units="kg/m**3",
            val=8960,
            desc="High density polyethylene for cable sheath",
        )

        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness",
            units="m",
            val=0.2e-2,
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
        t_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness"]

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]
        rho_in = inputs["settings:propulsion:he_power_train:DC_cable_harness:insulation:density"]
        rho_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density"
        ]
        rho_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:density"]

        m_conductor = np.pi * rho_c * (r_c ** 2.0)
        m_i = np.pi * rho_in * ((2.0 * r_c + t_in) * t_in)
        m_shield = np.pi * rho_shield * ((2.0 * (r_c + t_in) + t_shield) * t_shield)
        m_sheath = np.pi * rho_sheath * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath)

        m_cable = m_conductor + m_i + m_shield + m_sheath

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
        t_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness"]

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
        ] = np.pi * (r_c ** 2.0)
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:density",
        ] = np.pi * ((2.0 * r_c + t_in) * t_in)
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:density",
        ] = np.pi * ((2.0 * (r_c + t_in) + t_shield) * t_shield)
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:density",
        ] = np.pi * ((2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath)

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = (
            2.0
            * np.pi
            * (rho_c * r_c + rho_in * t_in + rho_shield * t_shield + rho_sheath * t_sheath)
        )
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = (
            2.0 * np.pi * (rho_in * (t_in + r_c) + rho_shield * t_shield + rho_sheath * t_sheath)
        )
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = (
            2.0 * np.pi * (rho_shield * (t_shield + r_c + t_in) + rho_sheath * t_sheath)
        )
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness",
        ] = (
            2.0 * np.pi * rho_sheath * (r_c + t_in + t_shield + t_sheath)
        )
