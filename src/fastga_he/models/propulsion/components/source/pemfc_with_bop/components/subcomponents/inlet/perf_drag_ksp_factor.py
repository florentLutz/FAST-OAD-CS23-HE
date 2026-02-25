# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDragKspFactor(om.ExplicitComponent):
    """
    Computation of the Ksp factor in the inlet drag coefficient.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "air_mass_flow_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "k_sp_factor",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]

        if design_mach == 0.9 and air_mass_flow_ratio <= 1.0:
            ksp = 0.6648 * air_mass_flow_ratio**2.0 - 1.6882 * air_mass_flow_ratio + 1.0151
        elif design_mach == 0.8 and air_mass_flow_ratio <= 0.656:
            ksp = 1.6992 * air_mass_flow_ratio**2.0 - 2.6385 * air_mass_flow_ratio + 1.0
        elif design_mach == 0.7 and air_mass_flow_ratio <= 0.4992:
            ksp = 1.904 * air_mass_flow_ratio**2.0 - 2.96 * air_mass_flow_ratio + 1.0027
        elif design_mach == 0.55 and air_mass_flow_ratio <= 0.4096:
            ksp = 2.432 * air_mass_flow_ratio**2.0 - 3.4359 * air_mass_flow_ratio + 1.0005
        elif design_mach == 0.2 and air_mass_flow_ratio <= 0.254933:
            ksp = 8.4724 * air_mass_flow_ratio**2.0 - 6.1278 * air_mass_flow_ratio + 1.0047
        elif 0.8 < design_mach < 0.9 and air_mass_flow_ratio < 1.0:
            factor = (0.9 - design_mach) / 0.1
            ksp = factor * (
                1.6992 * air_mass_flow_ratio**2.0 - 2.6385 * air_mass_flow_ratio + 1.0
            ) + (1.0 - factor) * (
                0.6648 * air_mass_flow_ratio**2.0 - 1.6882 * air_mass_flow_ratio + 1.0151
            )
        elif 0.7 < design_mach < 0.8 and air_mass_flow_ratio < 0.656:
            factor = (0.8 - design_mach) / 0.1
            ksp = factor * (
                1.904 * air_mass_flow_ratio**2.0 - 2.96 * air_mass_flow_ratio + 1.0027
            ) + (1.0 - factor) * (
                1.6992 * air_mass_flow_ratio**2.0 - 2.6385 * air_mass_flow_ratio + 1.0
            )
        elif 0.55 < design_mach < 0.7 and air_mass_flow_ratio < 0.4992:
            factor = (0.7 - design_mach) / (0.7 - 0.55)
            ksp = factor * (
                2.432 * air_mass_flow_ratio**2.0 - 3.4359 * air_mass_flow_ratio + 1.0005
            ) + (1.0 - factor) * (
                1.904 * air_mass_flow_ratio**2.0 - 2.96 * air_mass_flow_ratio + 1.0027
            )
        elif 0.2 < design_mach < 0.55 and air_mass_flow_ratio < 0.4096:
            factor = (0.55 - design_mach) / (0.55 - 0.2)
            ksp = factor * (
                8.4724 * air_mass_flow_ratio**2 - 6.1278 * air_mass_flow_ratio + 1.0047
            ) + (1 - factor) * (
                2.432 * air_mass_flow_ratio**2 - 3.4359 * air_mass_flow_ratio + 1.0005
            )
        else:
            ksp = 0.0

        outputs["k_sp_factor"] = ksp

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]

        if design_mach == 0.9 and air_mass_flow_ratio <= 1.0:
            dk_dm = 0.0
            dk_da = 1.3296 * air_mass_flow_ratio - 1.6882
        elif design_mach == 0.8 and air_mass_flow_ratio <= 0.656:
            dk_dm = 0.0
            dk_da = 3.3984 * air_mass_flow_ratio - 2.638
        elif design_mach == 0.7 and air_mass_flow_ratio <= 0.4992:
            dk_dm = 0.0
            dk_da = 3.808 * air_mass_flow_ratio - 2.96
        elif design_mach == 0.55 and air_mass_flow_ratio <= 0.4096:
            dk_dm = 0.0
            dk_da = 4.864 * air_mass_flow_ratio - 3.4359
        elif design_mach == 0.2 and air_mass_flow_ratio <= 0.254933:
            dk_dm = 0.0
            dk_da = 16.9448 * air_mass_flow_ratio - 6.1278
        elif 0.8 < design_mach < 0.9 and air_mass_flow_ratio < 1.0:
            dk_dm = -1.0 / 0.1 * (
                1.6992 * air_mass_flow_ratio**2.0 - 2.6385 * air_mass_flow_ratio + 1.0
            ) + (
                1.0
                / 0.1
                * (0.6648 * air_mass_flow_ratio**2.0 - 1.6882 * air_mass_flow_ratio + 1.0151)
            )
            dk_da = (0.9 - design_mach) / 0.1 * (3.3984 * air_mass_flow_ratio - 2.638) + (
                1.0 - (0.9 - design_mach) / 0.1
            ) * (1.3296 * air_mass_flow_ratio - 1.6882)
        elif 0.7 < design_mach < 0.8 and air_mass_flow_ratio < 0.656:
            dk_dm = -1.0 / 0.1 * (
                1.904 * air_mass_flow_ratio**2.0 - 2.96 * air_mass_flow_ratio + 1.0027
            ) + 1.0 / 0.1 * (1.6992 * air_mass_flow_ratio**2.0 - 2.6385 * air_mass_flow_ratio + 1.0)
            dk_da = (0.8 - design_mach) / 0.1 * (3.808 * air_mass_flow_ratio - 2.96) + (
                1.0 - (0.8 - design_mach) / 0.1
            ) * (3.3984 * air_mass_flow_ratio - 2.6385)
        elif 0.55 < design_mach < 0.7 and air_mass_flow_ratio < 0.4992:
            dk_dm = -1.0 / (0.7 - 0.55) * (
                2.432 * air_mass_flow_ratio**2.0 - 3.4359 * air_mass_flow_ratio + 1.0005
            ) + 1.0 / (0.7 - 0.55) * (
                1.904 * air_mass_flow_ratio**2.0 - 2.96 * air_mass_flow_ratio + 1.0027
            )
            dk_da = (0.7 - design_mach) / (0.7 - 0.55) * (4.864 * air_mass_flow_ratio - 3.4359) + (
                1.0 - (0.7 - design_mach) / (0.7 - 0.55)
            ) * (3.808 * air_mass_flow_ratio - 2.96)
        elif 0.2 < design_mach < 0.55 and air_mass_flow_ratio < 0.4096:
            dk_dm = -1.0 / (0.55 - 0.2) * (
                8.4724 * air_mass_flow_ratio**2.0 - 6.1278 * air_mass_flow_ratio + 1.0047
            ) + 1.0 / (0.55 - 0.2) * (
                2.432 * air_mass_flow_ratio**2.0 - 3.4359 * air_mass_flow_ratio + 1.0005
            )
            dk_da = (0.55 - design_mach) / (0.55 - 0.2) * (
                16.9448 * air_mass_flow_ratio - 6.1278
            ) + (1.0 - (0.55 - design_mach) / (0.55 - 0.2)) * (4.864 * air_mass_flow_ratio - 3.4359)
        else:
            dk_dm = 0.0
            dk_da = 0.0

        partials[
            "k_sp_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
        ] = dk_dm
        partials["k_sp_factor", "air_mass_flow_ratio"] = dk_da
