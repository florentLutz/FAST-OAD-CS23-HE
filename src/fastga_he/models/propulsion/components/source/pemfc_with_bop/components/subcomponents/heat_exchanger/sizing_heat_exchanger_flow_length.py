# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatExchangerFlowLength(om.Group):
    """
    Sizes the air and coolant flow lengths by minimizing HEX volume subject
    to the UA requirement. SLSQP runs inside a SubmodelComp, fully isolated
    from the parent Problem driver. All variables are promoted transparently
    so sibling components consume coolant_flow_length, air_flow_length,
    HEX_volume, etc. as normal inputs.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_flow_length_lower",
            default=0.05,
            desc="Lower bound for coolant flow length (m)",
        )
        self.options.declare(
            name="coolant_flow_length_upper",
            default=0.5,
            desc="Upper bound for coolant flow length (m)",
        )
        self.options.declare(
            name="air_flow_length_lower",
            default=0.05,
            desc="Lower bound for air flow length (m)",
        )
        self.options.declare(
            name="air_flow_length_upper",
            default=0.5,
            desc="Upper bound for air flow length (m)",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        inner_group = _HEXSizingFlowLengthInnerGroup(
            pemfc_stack_bop_id=pemfc_stack_bop_id,
            coolant_flow_length_lower=self.options["coolant_flow_length_lower"],
            coolant_flow_length_upper=self.options["coolant_flow_length_upper"],
            air_flow_length_lower=self.options["air_flow_length_lower"],
            air_flow_length_upper=self.options["air_flow_length_upper"],
        )

        # Build the inner Problem that SubmodelComp requires.
        # The driver and model are attached to this inner Problem, which is
        # then passed as the sole argument to SubmodelComp.
        inner_prob = om.Problem()
        inner_prob.model = inner_group
        inner_prob.driver = om.ScipyOptimizeDriver()
        inner_prob.driver.options["optimizer"] = "SLSQP"
        inner_prob.driver.options["tol"] = 1e-6
        inner_prob.driver.options["maxiter"] = 100

        self.add_subsystem(
            name="hex_sizing",
            subsys=om.SubmodelComp(
                problem=inner_prob,
                inputs=[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:no_flow_length",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:UA",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:transfer_area_volume_ratio",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:plate_thickness",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:plate_thermally_conductivity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:fin_area_total_surface_ratio",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_coolant_dynamic_viscosity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_coolant_prandtl_number",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_coolant_thermal_conductivity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_dynamic_viscosity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_prandtl_number",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:mean_air_thermal_conductivity",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":coolant:mass_flow_rate",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_flow_rate",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:fin_hydraulic_diameter",
                    "separating_plate_count",
                ],
                outputs=[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:air_flow_length",
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":heat_exchanger:coolant_flow_length",
                    "HEX_volume",
                    "plate_area",
                    "total_transfer_area",
                    "UA_difference",
                ],
            ),
            promotes=["*"],
        )


class _HEXSizingFlowLengthInnerGroup(om.Group):
    """
    Inner group containing all HEX physics subsystems. Registers design
    variables, UA equality constraint, and HEX_volume objective so that
    SubmodelComp can run SLSQP in full isolation from the parent driver.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_flow_length_lower",
            default=0.05,
            desc="Lower bound for coolant flow length (m)",
        )
        self.options.declare(
            name="coolant_flow_length_upper",
            default=0.5,
            desc="Upper bound for coolant flow length (m)",
        )
        self.options.declare(
            name="air_flow_length_lower",
            default=0.05,
            desc="Lower bound for air flow length (m)",
        )
        self.options.declare(
            name="air_flow_length_upper",
            default=0.5,
            desc="Upper bound for air flow length (m)",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        # IndepVarComp owns the design variables so SLSQP has a proper
        # output to drive. Without this, coolant_flow_length and
        # air_flow_length have no source in the model and the optimizer
        # cannot update them.
        ivc = om.IndepVarComp()
        ivc.add_output("coolant_flow_length", units="m", val=0.1)
        ivc.add_output("air_flow_length", units="m", val=0.1)
        self.add_subsystem(name="flow_length_ivc", subsys=ivc, promotes=["*"])

        self.add_subsystem(
            name="hex_volume",
            subsys=_HEXVolume(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="plate_area",
            subsys=_PlateArea(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_transfer_area",
            subsys=_TotalTransferArea(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="reynolds_number",
            subsys=_ReynoldsNumber(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="colburn_number",
            subsys=_ColburnNumber(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="heat_transfer_coefficient",
            subsys=_HeatTransferCoefficient(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dimensionless_intermediate_variables",
            subsys=_DimensionlessIntermediateFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fin_effficiency",
            subsys=_FinEfficiency(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="overall_efficiency",
            subsys=_OverallSurfaceEfficiency(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_UA",
            subsys=_UADifference(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="flow_length_output",
            subsys=_FlowLength(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )

        # Both flow lengths are free design variables within physical bounds
        self.add_design_var(
            "coolant_flow_length",
            lower=self.options["coolant_flow_length_lower"],
            upper=self.options["coolant_flow_length_upper"],
            units="m",
        )
        self.add_design_var(
            "air_flow_length",
            lower=self.options["air_flow_length_lower"],
            upper=self.options["air_flow_length_upper"],
            units="m",
        )

        # Equality constraint: UA_difference must be zero (calculated == required)
        self.add_constraint("UA_difference", equals=0.0, units="W/K", alias="UA_residual")

        # Objective: find the smallest HEX that satisfies the UA constraint
        self.add_objective("HEX_volume", units="m**3")


class _HEXVolume(om.ExplicitComponent):
    """
    Computation of the volume of the cross-flow heat exchanger.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length",
            units="m",
            val=0.3,
        )
        self.add_input(
            name="coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="HEX_volume",
            units="m**3",
            val=0.3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length"
        ]

        outputs["HEX_volume"] = no_flow_length * coolant_flow_length * air_flow_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length"
        ]

        partials["HEX_volume", "coolant_flow_length"] = no_flow_length * air_flow_length

        partials["HEX_volume", "air_flow_length"] = no_flow_length * coolant_flow_length

        partials[
            "HEX_volume",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length",
        ] = coolant_flow_length * air_flow_length


class _PlateArea(om.ExplicitComponent):
    """
    Computation of the plate area.
    """

    def setup(self):
        self.add_input(
            name="separating_plate_count",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="plate_area",
            units="m**2",
            val=0.3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        separating_plate_count = inputs["separating_plate_count"]
        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]

        outputs["plate_area"] = separating_plate_count * coolant_flow_length * air_flow_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        separating_plate_count = inputs["separating_plate_count"]
        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]

        partials["plate_area", "separating_plate_count"] = coolant_flow_length * air_flow_length

        partials["plate_area", "coolant_flow_length"] = separating_plate_count * air_flow_length

        partials["plate_area", "air_flow_length"] = separating_plate_count * coolant_flow_length


class _TotalTransferArea(om.ExplicitComponent):
    """
    Computation of the total transfer area for both flow.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            units="1/m",
            val=np.nan,
        )
        self.add_input(
            name="HEX_volume",
            units="m**3",
            val=np.nan,
        )

        self.add_output(
            name="total_transfer_area",
            units="m**2",
            val=0.3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["total_transfer_area"] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:transfer_area_volume_ratio"
            ]
            * inputs["HEX_volume"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        partials[
            "total_transfer_area",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
        ] = inputs["HEX_volume"]

        partials["total_transfer_area", "HEX_volume"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio"
        ]


class _ReynoldsNumber(om.ExplicitComponent):
    """
    Computation of the Reynold's number for both flow.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
            units="1/m",
            val=np.nan,
        )
        self.add_input(
            name="coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            units="kg/s",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_rate",
            units="kg/s",
            val=np.nan,
        )

        self.add_output(
            name="air_reynolds_number",
            units="unitless",
            val=0.3,
        )
        self.add_output(
            name="coolant_reynolds_number",
            units="unitless",
            val=0.3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:transfer_area_volume_ratio",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:no_flow_length",
            ],
            method="exact",
        )

        self.declare_partials(
            "air_reynolds_number",
            [
                "air_flow_length",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_air_dynamic_viscosity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:air_flow_rate",
            ],
            method="exact",
        )

        self.declare_partials(
            "coolant_reynolds_number",
            [
                "coolant_flow_length",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_coolant_dynamic_viscosity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:mass_flow_rate",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length"
        ]
        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]
        mean_coolant_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_dynamic_viscosity"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_dynamic_viscosity"
        ]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_rate"
        ]
        transfer_area_volume_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio"
        ]

        outputs["air_reynolds_number"] = (4.0 * air_mass_flow_rate) / (
            transfer_area_volume_ratio
            * air_flow_length
            * no_flow_length
            * mean_air_dynamic_viscosity
        )

        outputs["coolant_reynolds_number"] = (4.0 * coolant_mass_flow_rate) / (
            transfer_area_volume_ratio
            * coolant_flow_length
            * no_flow_length
            * mean_coolant_dynamic_viscosity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length"
        ]
        coolant_flow_length = inputs["coolant_flow_length"]
        air_flow_length = inputs["air_flow_length"]
        mean_coolant_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_dynamic_viscosity"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_dynamic_viscosity"
        ]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_rate"
        ]
        transfer_area_volume_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio"
        ]

        common_air_denominator = (
            transfer_area_volume_ratio
            * air_flow_length
            * no_flow_length
            * mean_air_dynamic_viscosity
        )
        common_coolant_denominator = (
            transfer_area_volume_ratio
            * coolant_flow_length
            * no_flow_length
            * mean_coolant_dynamic_viscosity
        )

        partials["air_reynolds_number", "air_flow_length"] = -(4.0 * air_mass_flow_rate) / (
            common_air_denominator * air_flow_length
        )

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_dynamic_viscosity",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * mean_air_dynamic_viscosity)

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_rate",
        ] = 4.0 / common_air_denominator

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * no_flow_length)

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * transfer_area_volume_ratio)

        partials["coolant_reynolds_number", "coolant_flow_length"] = -(
            4.0 * coolant_mass_flow_rate
        ) / (common_coolant_denominator * coolant_flow_length)

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:transfer_area_volume_ratio",
        ] = -(4.0 * coolant_mass_flow_rate) / (
            transfer_area_volume_ratio * common_coolant_denominator
        )

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_dynamic_viscosity",
        ] = -(4.0 * coolant_mass_flow_rate) / (
            common_coolant_denominator * mean_coolant_dynamic_viscosity
        )

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
        ] = 4.0 / common_coolant_denominator

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:no_flow_length",
        ] = -(4.0 * coolant_mass_flow_rate) / (common_coolant_denominator * no_flow_length)


class _ColburnNumber(om.ExplicitComponent):
    """
    Computation of the Colburn number for both flow.
    """

    def setup(self):
        self.add_input(
            name="air_reynolds_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_reynolds_number",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            name="air_colburn_number",
            units="unitless",
            val=0.3,
        )
        self.add_output(
            name="coolant_colburn_number",
            units="unitless",
            val=0.3,
        )

    def setup_partials(self):
        self.declare_partials("air_colburn_number", "air_reynolds_number", method="exact")
        self.declare_partials("coolant_colburn_number", "coolant_reynolds_number", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]

        outputs["air_colburn_number"] = np.where(
            air_reynolds_number < 1500,
            0.52 * air_reynolds_number**-0.51,
            0.41 * air_reynolds_number**-0.46,
        )
        outputs["coolant_colburn_number"] = np.where(
            coolant_reynolds_number < 1500,
            0.52 * coolant_reynolds_number**-0.51,
            0.41 * coolant_reynolds_number**-0.46,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]

        partials["air_colburn_number", "air_reynolds_number"] = np.where(
            air_reynolds_number < 1500,
            -0.2652 * air_reynolds_number**-1.51,
            -0.1886 * air_reynolds_number**-1.46,
        )
        partials["coolant_colburn_number", "coolant_reynolds_number"] = np.where(
            coolant_reynolds_number < 1500,
            -0.2652 * coolant_reynolds_number**-1.51,
            -0.1886 * coolant_reynolds_number**-1.46,
        )


class _HeatTransferCoefficient(om.ExplicitComponent):
    """
    Computation of the heat transfer coefficient for both flow.
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
            name="air_colburn_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_colburn_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="air_reynolds_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_reynolds_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_prandtl_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_prandtl_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_thermal_conductivity",
            units="W/m/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_thermal_conductivity",
            units="W/m/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="air_heat_transfer_coefficient",
            units="W/m**2/K",
            val=0.3,
        )
        self.add_output(
            name="coolant_heat_transfer_coefficient",
            units="W/m**2/K",
            val=0.3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter",
            method="exact",
        )
        self.declare_partials(
            "air_heat_transfer_coefficient",
            [
                "air_colburn_number",
                "air_reynolds_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_air_prandtl_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_air_thermal_conductivity",
            ],
            method="exact",
        )
        self.declare_partials(
            "coolant_heat_transfer_coefficient",
            [
                "coolant_colburn_number",
                "coolant_reynolds_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_coolant_prandtl_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:mean_coolant_thermal_conductivity",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_colburn_number = inputs["air_colburn_number"]
        coolant_colburn_number = inputs["coolant_colburn_number"]
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]
        mean_air_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_prandtl_number"
        ]
        mean_coolant_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_prandtl_number"
        ]
        mean_air_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_thermal_conductivity"
        ]
        mean_coolant_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_thermal_conductivity"
        ]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter"
        ]

        outputs["air_heat_transfer_coefficient"] = (
            air_colburn_number
            * air_reynolds_number
            * mean_air_prandtl_number ** (1.0 / 3.0)
            * mean_air_thermal_conductivity
            / fin_hydraulic_diameter
        )
        outputs["coolant_heat_transfer_coefficient"] = (
            coolant_colburn_number
            * coolant_reynolds_number
            * mean_coolant_prandtl_number ** (1.0 / 3.0)
            * mean_coolant_thermal_conductivity
            / fin_hydraulic_diameter
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_colburn_number = inputs["air_colburn_number"]
        coolant_colburn_number = inputs["coolant_colburn_number"]
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]
        mean_air_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_prandtl_number"
        ]
        mean_coolant_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_prandtl_number"
        ]
        mean_air_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_thermal_conductivity"
        ]
        mean_coolant_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_thermal_conductivity"
        ]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter"
        ]

        partials["air_heat_transfer_coefficient", "air_colburn_number"] = (
            air_reynolds_number
            * mean_air_prandtl_number ** (1.0 / 3.0)
            * mean_air_thermal_conductivity
            / fin_hydraulic_diameter
        )

        partials["air_heat_transfer_coefficient", "air_reynolds_number"] = (
            air_colburn_number
            * mean_air_prandtl_number ** (1.0 / 3.0)
            * mean_air_thermal_conductivity
            / fin_hydraulic_diameter
        )

        partials[
            "air_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_prandtl_number",
        ] = (
            air_colburn_number
            * air_reynolds_number
            * mean_air_prandtl_number ** (-2.0 / 3.0)
            * mean_air_thermal_conductivity
            / (fin_hydraulic_diameter * 3.0)
        )

        partials[
            "air_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_air_thermal_conductivity",
        ] = (
            air_colburn_number
            * air_reynolds_number
            * mean_air_prandtl_number ** (1.0 / 3.0)
            / fin_hydraulic_diameter
        )

        partials[
            "air_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter",
        ] = -(
            air_colburn_number
            * air_reynolds_number
            * mean_air_prandtl_number ** (1.0 / 3.0)
            * mean_air_thermal_conductivity
            / fin_hydraulic_diameter**2.0
        )

        partials["coolant_heat_transfer_coefficient", "coolant_colburn_number"] = (
            coolant_reynolds_number
            * mean_coolant_prandtl_number ** (1.0 / 3.0)
            * mean_coolant_thermal_conductivity
            / fin_hydraulic_diameter
        )

        partials["coolant_heat_transfer_coefficient", "coolant_reynolds_number"] = (
            coolant_colburn_number
            * mean_coolant_prandtl_number ** (1.0 / 3.0)
            * mean_coolant_thermal_conductivity
            / fin_hydraulic_diameter
        )

        partials[
            "coolant_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_prandtl_number",
        ] = (
            coolant_colburn_number
            * coolant_reynolds_number
            * mean_coolant_prandtl_number ** (-2.0 / 3.0)
            * mean_coolant_thermal_conductivity
            / (fin_hydraulic_diameter * 3.0)
        )

        partials[
            "coolant_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:mean_coolant_thermal_conductivity",
        ] = (
            coolant_colburn_number
            * coolant_reynolds_number
            * mean_coolant_prandtl_number ** (1.0 / 3.0)
            / fin_hydraulic_diameter
        )

        partials[
            "coolant_heat_transfer_coefficient",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_hydraulic_diameter",
        ] = -(
            coolant_colburn_number
            * coolant_reynolds_number
            * mean_coolant_prandtl_number ** (1.0 / 3.0)
            * mean_coolant_thermal_conductivity
            / fin_hydraulic_diameter**2.0
        )


class _DimensionlessIntermediateFactor(om.ExplicitComponent):
    """
    Computation of an intermediate dimensionless factor used for the computation of the flow lengths.
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
            name="air_heat_transfer_coefficient",
            units="W/m**2/K",
            val=np.nan,
        )
        self.add_input(
            name="coolant_heat_transfer_coefficient",
            units="W/m**2/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thermal_conductivity",
            units="W/m/K",
            val=237.0,
            desc="The thermal conductivity of the fin material, which is typically aluminum",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
            units="m",
            val=1.02e-4,
        )

        self.add_output(
            name="air_dimensionless_intermediate_factor",
            units="unitless",
            val=0.3,
        )
        self.add_output(
            name="coolant_dimensionless_intermediate_factor",
            units="unitless",
            val=0.3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_thermal_conductivity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:fin_thickness",
            ],
            method="exact",
        )
        self.declare_partials(
            "air_dimensionless_intermediate_factor",
            "air_heat_transfer_coefficient",
            method="exact",
        )
        self.declare_partials(
            "coolant_dimensionless_intermediate_factor",
            "coolant_heat_transfer_coefficient",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        fin_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thermal_conductivity"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness"
        ]

        outputs["air_dimensionless_intermediate_factor"] = np.sqrt(
            2.0 * air_heat_transfer_coefficient / (fin_thermal_conductivity * fin_thickness)
        )
        outputs["coolant_dimensionless_intermediate_factor"] = np.sqrt(
            2.0 * coolant_heat_transfer_coefficient / (fin_thermal_conductivity * fin_thickness)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        fin_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thermal_conductivity"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness"
        ]

        partials["air_dimensionless_intermediate_factor", "air_heat_transfer_coefficient"] = (
            1.0
            / np.sqrt(
                2.0 * fin_thermal_conductivity * fin_thickness * air_heat_transfer_coefficient
            )
        )

        partials[
            "air_dimensionless_intermediate_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thermal_conductivity",
        ] = -(
            air_heat_transfer_coefficient
            / np.sqrt(
                2.0 * fin_thermal_conductivity**3.0 * fin_thickness * air_heat_transfer_coefficient
            )
        )

        partials[
            "air_dimensionless_intermediate_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
        ] = -(
            air_heat_transfer_coefficient
            / np.sqrt(
                2.0 * fin_thermal_conductivity * fin_thickness**3.0 * air_heat_transfer_coefficient
            )
        )

        partials[
            "coolant_dimensionless_intermediate_factor", "coolant_heat_transfer_coefficient"
        ] = 1.0 / np.sqrt(
            2.0 * fin_thermal_conductivity * fin_thickness * coolant_heat_transfer_coefficient
        )

        partials[
            "coolant_dimensionless_intermediate_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thermal_conductivity",
        ] = -(
            coolant_heat_transfer_coefficient
            / np.sqrt(
                2.0
                * fin_thermal_conductivity**3.0
                * fin_thickness
                * coolant_heat_transfer_coefficient
            )
        )

        partials[
            "coolant_dimensionless_intermediate_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_thickness",
        ] = -(
            coolant_heat_transfer_coefficient
            / np.sqrt(
                2.0
                * fin_thermal_conductivity
                * fin_thickness**3.0
                * coolant_heat_transfer_coefficient
            )
        )


class _FinEfficiency(om.ExplicitComponent):
    """
    Computation of the fin efficiency for both flow.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            units="m",
            val=6.25e-3,
        )
        self.add_input(
            name="air_dimensionless_intermediate_factor",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_dimensionless_intermediate_factor",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            name="air_fin_efficiency",
            units="unitless",
            val=0.3,
        )
        self.add_output(
            name="coolant_fin_efficiency",
            units="unitless",
            val=0.3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
            method="exact",
        )
        self.declare_partials(
            "air_fin_efficiency",
            "air_dimensionless_intermediate_factor",
            method="exact",
        )
        self.declare_partials(
            "coolant_fin_efficiency",
            "coolant_dimensionless_intermediate_factor",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height"
        ]
        air_dimensionless_intermediate_factor = inputs["air_dimensionless_intermediate_factor"]
        coolant_dimensionless_intermediate_factor = inputs[
            "coolant_dimensionless_intermediate_factor"
        ]

        outputs["air_fin_efficiency"] = np.tanh(
            0.5 * air_dimensionless_intermediate_factor * fin_height
        ) / (0.5 * air_dimensionless_intermediate_factor * fin_height)
        outputs["coolant_fin_efficiency"] = np.tanh(
            0.5 * coolant_dimensionless_intermediate_factor * fin_height
        ) / (0.5 * coolant_dimensionless_intermediate_factor * fin_height)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height"
        ]
        air_dimensionless_intermediate_factor = inputs["air_dimensionless_intermediate_factor"]
        coolant_dimensionless_intermediate_factor = inputs[
            "coolant_dimensionless_intermediate_factor"
        ]

        partials[
            "air_fin_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
        ] = np.cosh(
            0.5 * air_dimensionless_intermediate_factor * fin_height
        ) ** -2.0 / fin_height - (
            2.0 * np.tanh(0.5 * air_dimensionless_intermediate_factor * fin_height)
        ) / (air_dimensionless_intermediate_factor * fin_height**2.0)

        partials["air_fin_efficiency", "air_dimensionless_intermediate_factor"] = np.cosh(
            0.5 * air_dimensionless_intermediate_factor * fin_height
        ) ** -2.0 / air_dimensionless_intermediate_factor - (
            2.0 * np.tanh(0.5 * air_dimensionless_intermediate_factor * fin_height)
        ) / (air_dimensionless_intermediate_factor**2.0 * fin_height)

        partials[
            "coolant_fin_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_height",
        ] = np.cosh(
            0.5 * coolant_dimensionless_intermediate_factor * fin_height
        ) ** -2.0 / fin_height - (
            2.0 * np.tanh(0.5 * coolant_dimensionless_intermediate_factor * fin_height)
        ) / (coolant_dimensionless_intermediate_factor * fin_height**2.0)

        partials["coolant_fin_efficiency", "coolant_dimensionless_intermediate_factor"] = np.cosh(
            0.5 * coolant_dimensionless_intermediate_factor * fin_height
        ) ** -2.0 / coolant_dimensionless_intermediate_factor - (
            2.0 * np.tanh(0.5 * coolant_dimensionless_intermediate_factor * fin_height)
        ) / (coolant_dimensionless_intermediate_factor**2.0 * fin_height)


class _OverallSurfaceEfficiency(om.ExplicitComponent):
    """
    Computation of the overall surface efficiency for both flow.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio",
            units="unitless",
            val=0.809,
        )
        self.add_input(
            name="air_fin_efficiency",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_fin_efficiency",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            name="air_overall_surface_efficiency",
            units="unitless",
            val=0.3,
        )
        self.add_output(
            name="coolant_overall_surface_efficiency",
            units="unitless",
            val=0.3,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio",
            method="exact",
        )
        self.declare_partials(
            "air_overall_surface_efficiency",
            "air_fin_efficiency",
            method="exact",
        )
        self.declare_partials(
            "coolant_overall_surface_efficiency",
            "coolant_fin_efficiency",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_area_total_surface_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio"
        ]
        air_fin_efficiency = inputs["air_fin_efficiency"]
        coolant_fin_efficiency = inputs["coolant_fin_efficiency"]

        outputs["air_overall_surface_efficiency"] = 1.0 - fin_area_total_surface_ratio * (
            1.0 - air_fin_efficiency
        )
        outputs["coolant_overall_surface_efficiency"] = 1.0 - fin_area_total_surface_ratio * (
            1.0 - coolant_fin_efficiency
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        fin_area_total_surface_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio"
        ]
        air_fin_efficiency = inputs["air_fin_efficiency"]
        coolant_fin_efficiency = inputs["coolant_fin_efficiency"]

        partials[
            "air_overall_surface_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio",
        ] = -(1.0 - air_fin_efficiency)

        partials[
            "air_overall_surface_efficiency",
            "air_fin_efficiency",
        ] = fin_area_total_surface_ratio

        partials[
            "coolant_overall_surface_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:fin_area_total_surface_ratio",
        ] = -(1.0 - coolant_fin_efficiency)

        partials[
            "coolant_overall_surface_efficiency",
            "coolant_fin_efficiency",
        ] = fin_area_total_surface_ratio


class _UADifference(om.ExplicitComponent):
    """
    Computation of the calculated UA and the difference with respect to the required UA.
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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:UA",
            units="W/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness",
            units="m",
            val=8e-4,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thermally_conductivity",
            units="W/m/K",
            val=237.0,
            desc="The thermal conductivity of the plate material, which is typically aluminum",
        )
        self.add_input(
            name="air_heat_transfer_coefficient",
            units="W/m**2/K",
            val=np.nan,
        )
        self.add_input(
            name="coolant_heat_transfer_coefficient",
            units="W/m**2/K",
            val=np.nan,
        )
        self.add_input(
            name="air_overall_surface_efficiency",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_overall_surface_efficiency",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="plate_area",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            name="total_transfer_area",
            units="m**2",
            val=np.nan,
        )

        self.add_output(
            name="UA_difference",
            units="W/K",
            val=0.3,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        required_UA = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:UA"
        ]
        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness"
        ]
        plate_thermally_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thermally_conductivity"
        ]
        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        air_overall_surface_efficiency = inputs["air_overall_surface_efficiency"]
        coolant_overall_surface_efficiency = inputs["coolant_overall_surface_efficiency"]
        plate_area = inputs["plate_area"]
        total_transfer_area = inputs["total_transfer_area"]

        calculated_UA = (
            (air_overall_surface_efficiency * air_heat_transfer_coefficient * total_transfer_area)
            ** -1.0
            + (
                coolant_overall_surface_efficiency
                * coolant_heat_transfer_coefficient
                * total_transfer_area
            )
            ** -1.0
            + plate_thickness / (plate_thermally_conductivity * plate_area)
        ) ** -1.0

        outputs["UA_difference"] = calculated_UA - required_UA

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness"
        ]
        plate_thermally_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thermally_conductivity"
        ]
        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        air_overall_surface_efficiency = inputs["air_overall_surface_efficiency"]
        coolant_overall_surface_efficiency = inputs["coolant_overall_surface_efficiency"]
        plate_area = inputs["plate_area"]
        total_transfer_area = inputs["total_transfer_area"]

        calculated_UA = (
            (air_overall_surface_efficiency * air_heat_transfer_coefficient * total_transfer_area)
            ** -1.0
            + (
                coolant_overall_surface_efficiency
                * coolant_heat_transfer_coefficient
                * total_transfer_area
            )
            ** -1.0
            + plate_thickness / (plate_thermally_conductivity * plate_area)
        ) ** -1.0

        partials[
            "UA_difference",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:UA",
        ] = -1.0

        partials[
            "UA_difference",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thickness",
        ] = -(calculated_UA**2.0) / (plate_thermally_conductivity * plate_area)

        partials[
            "UA_difference",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:plate_thermally_conductivity",
        ] = calculated_UA**2.0 * plate_thickness / (plate_thermally_conductivity**2.0 * plate_area)

        partials["UA_difference", "plate_area"] = (
            calculated_UA**2.0 * plate_thickness / (plate_thermally_conductivity * plate_area**2.0)
        )

        # dR/dAt = -1/(eta_oa*ha*At^2) - 1/(eta_oc*hc*At^2)
        # dUA/dAt = -(UA^2) * dR/dAt = (UA^2) * (1/(eta_oa*ha) + 1/(eta_oc*hc)) / At^2
        partials["UA_difference", "total_transfer_area"] = (
            calculated_UA**2.0
            * (
                (air_overall_surface_efficiency * air_heat_transfer_coefficient) ** -1.0
                + (coolant_overall_surface_efficiency * coolant_heat_transfer_coefficient) ** -1.0
            )
            / total_transfer_area**2.0
        )

        # dR/dha = -1/(eta_oa*ha^2*At)
        # dUA/dha = -(UA^2)*dR/dha = (UA^2)/(eta_oa*ha^2*At)
        partials["UA_difference", "air_heat_transfer_coefficient"] = calculated_UA**2.0 / (
            air_overall_surface_efficiency
            * air_heat_transfer_coefficient**2.0
            * total_transfer_area
        )

        # dR/dhc = -1/(eta_oc*hc^2*At)
        # dUA/dhc = (UA^2)/(eta_oc*hc^2*At)
        partials["UA_difference", "coolant_heat_transfer_coefficient"] = calculated_UA**2.0 / (
            coolant_overall_surface_efficiency
            * coolant_heat_transfer_coefficient**2.0
            * total_transfer_area
        )

        # dR/d(eta_oa) = -1/(eta_oa^2*ha*At)
        # dUA/d(eta_oa) = (UA^2)/(eta_oa^2*ha*At)
        partials["UA_difference", "air_overall_surface_efficiency"] = calculated_UA**2.0 / (
            air_overall_surface_efficiency**2.0
            * air_heat_transfer_coefficient
            * total_transfer_area
        )

        # dR/d(eta_oc) = -1/(eta_oc^2*hc*At)
        # dUA/d(eta_oc) = (UA^2)/(eta_oc^2*hc*At)
        partials["UA_difference", "coolant_overall_surface_efficiency"] = calculated_UA**2.0 / (
            coolant_overall_surface_efficiency**2.0
            * coolant_heat_transfer_coefficient
            * total_transfer_area
        )


class _FlowLength(om.ExplicitComponent):
    """
    Flow length output
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

        self.add_input("air_flow_length", units="m", val=np.nan)
        self.add_input("coolant_flow_length", units="m", val=np.nan)

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_length",
            units="m",
            val=0.1,
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:coolant_flow_length",
            units="m",
            val=0.05,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_length",
            "air_flow_length",
            val=1.0,
        )

        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:coolant_flow_length",
            "coolant_flow_length",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:air_flow_length"
        ] = inputs["air_flow_length"]
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:coolant_flow_length"
        ] = inputs["coolant_flow_length"]
