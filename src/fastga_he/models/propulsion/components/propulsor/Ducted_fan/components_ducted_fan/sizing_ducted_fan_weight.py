# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
sizing_ducted_fan_weight.py
============================

Scope: this component estimates ONLY the rotor (blades + hub) and duct mass
of a SINGLE ducted fan unit. Motor, ESC, and all other electrical propulsion
components have their own dedicated sizing modules elsewhere in the
FAST-OAD-CS23-HE power train and are NOT computed here, to avoid double
counting.

Mass breakdown (identical physics to edf_mass.py v4, just per unit instead
of an N_fans aggregate -- see that file's history for the calibration
rationale):
    mass_rotor = K_ROTOR * number_blades * diameter**N_EXP
    mass_duct  = K_DUCT  * pi * diameter * (CHORD_DUCT_RATIO * diameter)
    mass       = mass_rotor + mass_duct

blade_chord is NOT used in the rotor mass formula (no reliable chord data
existed for the two calibration points -- see edf_mass.py history). It
remains an input for interface compatibility and for possible use elsewhere
in the sizing chain (e.g. solidity/constraints), but does not affect mass.
"""

import numpy as np
import openmdao.api as om


# ── Rotor mass: empirical power law, fit to 2 real data points ────────────
# Schubeler DS-51-AXI HDS (90mm, 10 blades, 62g) and
# Schubeler DS-86-AXI HDS (120mm, 10 blades, 138g) -- see edf_mass.py history.
N_EXP = 2.78  # diameter exponent (fit) -- close to D^3, NOT D^1.5
K_ROTOR = 5.02  # kg / (blade * m^N_EXP)

# ── Duct (empirical) ────────────────────────────────────────────────────────
# Target: mass_duct = 0.40 kg at D=195mm
K_DUCT = 8.37  # kg / m^2  -- duct mass per unit lateral area
CHORD_DUCT_RATIO = 0.40  # duct chord / fan diameter (typical)


class SizingDuctedFanWeight(om.ExplicitComponent):
    """
    Rotor + duct mass estimation for a single ducted fan unit.
    Motor and ESC mass are NOT included -- see module docstring.
    """

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the ducted fan",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades",
            val=np.nan,
            desc="Number of blades on the ducted fan rotor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord",
            val=np.nan,
            units="m",
            desc="Blade chord of the ducted fan rotor (interface compatibility only, unused in "
            "the rotor mass formula, see module docstring)",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":mass",
            val=0.5,
            units="kg",
            desc="Mass of one ducted fan unit (rotor + duct, motor/ESC excluded)",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ]
        number_blades = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades"
        ]

        mass_rotor = K_ROTOR * number_blades * diameter**N_EXP
        mass_duct = K_DUCT * np.pi * diameter * (CHORD_DUCT_RATIO * diameter)

        outputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":mass"] = (
            mass_rotor + mass_duct
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ]
        number_blades = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades"
        ]

        d_mass_rotor_d_diameter = K_ROTOR * number_blades * N_EXP * diameter ** (N_EXP - 1.0)
        d_mass_duct_d_diameter = K_DUCT * np.pi * 2.0 * CHORD_DUCT_RATIO * diameter
        d_mass_rotor_d_nb = K_ROTOR * diameter**N_EXP

        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":mass",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
        ] = d_mass_rotor_d_diameter + d_mass_duct_d_diameter
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":mass",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades",
        ] = d_mass_rotor_d_nb
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":mass",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord",
        ] = 0.0
