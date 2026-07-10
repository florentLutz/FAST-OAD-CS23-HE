# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatExchangerFlowLength(om.Group):
    """
    Computes the coolant_flow_length of the heat exchanger by solving
    the UA residual equation implicitly, given that air_flow_length is
    provided by an external component.  No internal optimizer is needed:
    a Newton solver drives the single implicit unknown
    (coolant_flow_length) until UA_difference == 0.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_subsystem(
            name="coolant_flow_length",
            subsys=_CoolantFlowLength(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )

        # The implicit component that owns air_flow_length as a state
        # and uses UA_difference as its residual.
        self.add_subsystem(
            name="air_flow_length_implicit",
            subsys=_AirFlowLengthImplicit(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=heat_exchanger_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="hex_volume",
            subsys=_HEXVolume(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="plate_area",
            subsys=_PlateArea(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_transfer_area",
            subsys=_TotalTransferArea(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="reynolds_number",
            subsys=_ReynoldsNumber(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="colburn_number",
            subsys=_ColburnNumber(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="heat_transfer_coefficient",
            subsys=_HeatTransferCoefficient(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dimensionless_intermediate_variables",
            subsys=_DimensionlessIntermediateFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fin_effficiency",
            subsys=_FinEfficiency(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="overall_efficiency",
            subsys=_OverallSurfaceEfficiency(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_UA",
            subsys=_UADifference(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="flow_length_output",
            subsys=_FlowLengthOutput(
                pemfc_stack_bop_id=pemfc_stack_bop_id, heat_exchanger_id=heat_exchanger_id
            ),
            promotes=["*"],
        )

        # Newton solver to converge  air_flow_length so that
        # UA_difference == 0.  The group contains a cycle because
        #  air_flow_length_implicit outputs  air_flow_length which
        # feeds through the explicit chain back to UA_difference, which is
        # read by the implicit component.
        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        newton.options["rtol"] = 1e-4
        newton.options["maxiter"] = 10
        newton.options["iprint"] = 0
        self.nonlinear_solver.options["stall_limit"] = 5
        self.nonlinear_solver.options["stall_tol"] = 1e-5

        self.linear_solver = om.DirectSolver()

        # Bounds enforcement so Newton doesn't wander to negative lengths
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["print_bound_enforce"] = False


class _CoolantFlowLength(om.ExplicitComponent):
    """
    Computation of the air flow length, which is derived by the fraction between the flush_inlet area
    and the no flow length.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="air_flow_area",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":expansion_area_ratio",
            val=1.1,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
            units="m",
            val=0.1,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_flow_area = inputs["air_flow_area"]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]
        expansion_area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":expansion_area_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ] = np.clip(air_flow_area * expansion_area_ratio / no_flow_length, 0.05, np.inf)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_flow_area = inputs["air_flow_area"]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]
        expansion_area_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":expansion_area_ratio"
        ]

        coolant_flow_length = air_flow_area * expansion_area_ratio / no_flow_length
        clipped_coolant_flow_length = np.clip(coolant_flow_length, 0.05, np.inf)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
            "air_flow_area",
        ] = np.where(
            coolant_flow_length == clipped_coolant_flow_length,
            expansion_area_ratio / no_flow_length,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
        ] = np.where(
            coolant_flow_length == clipped_coolant_flow_length,
            -air_flow_area * expansion_area_ratio / no_flow_length**2.0,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":expansion_area_ratio",
        ] = np.where(
            coolant_flow_length == clipped_coolant_flow_length, air_flow_area / no_flow_length, 1e-6
        )


class _AirFlowLengthImplicit(om.ImplicitComponent):
    """
    Implicit component whose single state variable is  air_flow_length.
    The residual is UA_difference (calculated_UA - required_UA), which must
    equal zero.  All physics computations happen in the surrounding explicit
    components; this component simply reads the resulting UA_difference.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        self.add_input(name="UA_difference", units="W/K", val=0.0)

        self.add_output(name="air_flow_length", units="m", val=0.05)

    def setup_partials(self):
        self.declare_partials("air_flow_length", "UA_difference", val=1.0)
        self.declare_partials("air_flow_length", "air_flow_length", val=0.0)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        residuals["air_flow_length"] = inputs["UA_difference"]


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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_flow_length = np.clip(inputs["air_flow_length"], 0.05, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]

        outputs["HEX_volume"] = no_flow_length * coolant_flow_length * air_flow_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_flow_length = np.clip(inputs["air_flow_length"], 0.05, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]

        partials["HEX_volume", "air_flow_length"] = no_flow_length * coolant_flow_length

        partials[
            "HEX_volume",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
        ] = no_flow_length * air_flow_length

        partials[
            "HEX_volume",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
        ] = coolant_flow_length * air_flow_length


class _PlateArea(om.ExplicitComponent):
    """
    Computation of the plate area.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="separating_plate_count",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
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
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        separating_plate_count = inputs["separating_plate_count"]
        air_flow_length = np.clip(inputs["air_flow_length"], 0.05, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]

        outputs["plate_area"] = separating_plate_count * coolant_flow_length * air_flow_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        separating_plate_count = inputs["separating_plate_count"]
        air_flow_length = np.clip(inputs["air_flow_length"], 0.05, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]

        partials["plate_area", "separating_plate_count"] = coolant_flow_length * air_flow_length

        partials["plate_area", "air_flow_length"] = separating_plate_count * coolant_flow_length

        partials[
            "plate_area",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
        ] = separating_plate_count * air_flow_length


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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        outputs["total_transfer_area"] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":transfer_area_volume_ratio"
            ]
            * inputs["HEX_volume"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        partials[
            "total_transfer_area",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio",
        ] = inputs["HEX_volume"]

        partials["total_transfer_area", "HEX_volume"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio"
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio",
            units="1/m",
            val=np.nan,
        )
        self.add_input(
            name="air_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
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
            + ":"
            + heat_exchanger_id
            + ":air_flow_rate",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "*",
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":transfer_area_volume_ratio",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":no_flow_length",
            ],
            method="exact",
        )

        self.declare_partials(
            "air_reynolds_number",
            [
                "air_flow_length",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":mean_air_dynamic_viscosity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":air_flow_rate",
            ],
            method="exact",
        )

        self.declare_partials(
            "coolant_reynolds_number",
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":coolant_flow_length",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":mean_coolant_dynamic_viscosity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":coolant:mass_flow_rate",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]
        air_flow_length = np.clip(inputs["air_flow_length"], 0.01, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        mean_coolant_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_dynamic_viscosity"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity"
        ]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_rate"
        ]
        transfer_area_volume_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio"
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length"
        ]
        air_flow_length = np.clip(inputs["air_flow_length"], 0.01, 1.0)
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        mean_coolant_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_dynamic_viscosity"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity"
        ]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        air_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_rate"
        ]
        transfer_area_volume_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio"
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
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * mean_air_dynamic_viscosity)

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_rate",
        ] = 4.0 / common_air_denominator

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * no_flow_length)

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio",
        ] = -(4.0 * air_mass_flow_rate) / (common_air_denominator * transfer_area_volume_ratio)

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
        ] = -(4.0 * coolant_mass_flow_rate) / (common_coolant_denominator * coolant_flow_length)

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":transfer_area_volume_ratio",
        ] = -(4.0 * coolant_mass_flow_rate) / (
            transfer_area_volume_ratio * common_coolant_denominator
        )

        partials[
            "coolant_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_dynamic_viscosity",
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
            + ":"
            + heat_exchanger_id
            + ":no_flow_length",
        ] = -(4.0 * coolant_mass_flow_rate) / (common_coolant_denominator * no_flow_length)


class _ColburnNumber(om.ExplicitComponent):
    """
    Computation of the Colburn number for both flow. Surrogate model
    based on the Reynolds number of the flow, with a transition at 1500, obtained from
    Valentine's thesis.
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

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
            + ":"
            + heat_exchanger_id
            + ":mean_air_prandtl_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_prandtl_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_thermal_conductivity",
            units="W/m/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_thermal_conductivity",
            units="W/m/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            method="exact",
        )
        self.declare_partials(
            "air_heat_transfer_coefficient",
            [
                "air_colburn_number",
                "air_reynolds_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":mean_air_prandtl_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":mean_air_thermal_conductivity",
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
                + ":"
                + heat_exchanger_id
                + ":mean_coolant_prandtl_number",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":mean_coolant_thermal_conductivity",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_colburn_number = inputs["air_colburn_number"]
        coolant_colburn_number = inputs["coolant_colburn_number"]
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]
        mean_air_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_prandtl_number"
        ]
        mean_coolant_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_prandtl_number"
        ]
        mean_air_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_thermal_conductivity"
        ]
        mean_coolant_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_thermal_conductivity"
        ]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_colburn_number = inputs["air_colburn_number"]
        coolant_colburn_number = inputs["coolant_colburn_number"]
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]
        mean_air_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_prandtl_number"
        ]
        mean_coolant_prandtl_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_prandtl_number"
        ]
        mean_air_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_thermal_conductivity"
        ]
        mean_coolant_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_thermal_conductivity"
        ]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
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
            + ":"
            + heat_exchanger_id
            + ":mean_air_prandtl_number",
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
            + ":"
            + heat_exchanger_id
            + ":mean_air_thermal_conductivity",
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
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
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
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_prandtl_number",
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
            + ":"
            + heat_exchanger_id
            + ":mean_coolant_thermal_conductivity",
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
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

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
            + ":"
            + heat_exchanger_id
            + ":fin_thermal_conductivity",
            units="W/m/K",
            val=237.0,
            desc="The thermal conductivity of the fin material, which is typically aluminum",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "*",
            [
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":fin_thermal_conductivity",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":fin_thickness",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        fin_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thermal_conductivity"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness"
        ]

        outputs["air_dimensionless_intermediate_factor"] = np.sqrt(
            2.0 * air_heat_transfer_coefficient / (fin_thermal_conductivity * fin_thickness)
        )
        outputs["coolant_dimensionless_intermediate_factor"] = np.sqrt(
            2.0 * coolant_heat_transfer_coefficient / (fin_thermal_conductivity * fin_thickness)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_heat_transfer_coefficient = inputs["air_heat_transfer_coefficient"]
        coolant_heat_transfer_coefficient = inputs["coolant_heat_transfer_coefficient"]
        fin_thermal_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thermal_conductivity"
        ]
        fin_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_thickness"
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
            + ":"
            + heat_exchanger_id
            + ":fin_thermal_conductivity",
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
            + ":"
            + heat_exchanger_id
            + ":fin_thickness",
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
            + ":"
            + heat_exchanger_id
            + ":fin_thermal_conductivity",
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
            + ":"
            + heat_exchanger_id
            + ":fin_thickness",
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height"
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_height = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height"
        ]
        air_dimensionless_intermediate_factor = inputs["air_dimensionless_intermediate_factor"]
        coolant_dimensionless_intermediate_factor = inputs[
            "coolant_dimensionless_intermediate_factor"
        ]

        partials[
            "air_fin_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_height",
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
            + ":"
            + heat_exchanger_id
            + ":fin_height",
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_area_total_surface_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio"
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        fin_area_total_surface_ratio = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio"
        ]
        air_fin_efficiency = inputs["air_fin_efficiency"]
        coolant_fin_efficiency = inputs["coolant_fin_efficiency"]

        partials[
            "air_overall_surface_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio",
        ] = -(1.0 - air_fin_efficiency)

        partials[
            "air_overall_surface_efficiency",
            "air_fin_efficiency",
        ] = fin_area_total_surface_ratio

        partials[
            "coolant_overall_surface_efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_area_total_surface_ratio",
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
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":UA",
            units="W/K",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thickness",
            units="m",
            val=8e-4,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thermally_conductivity",
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        required_UA = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":UA"
        ]
        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thickness"
        ]
        plate_thermally_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thermally_conductivity"
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
        heat_exchanger_id = self.options["heat_exchanger_id"]

        plate_thickness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thickness"
        ]
        plate_thermally_conductivity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thermally_conductivity"
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
            + ":"
            + heat_exchanger_id
            + ":UA",
        ] = -1.0

        partials[
            "UA_difference",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thickness",
        ] = -(calculated_UA**2.0) / (plate_thermally_conductivity * plate_area)

        partials[
            "UA_difference",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":plate_thermally_conductivity",
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


class _FlowLengthOutput(om.ExplicitComponent):
    """
    Copies coolant_flow_length to the properly-named output variable.
    air_flow_length is now an input from an external component, so it is
    simply passed through.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.add_input("air_flow_length", units="m", val=np.nan)

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length",
            units="m",
            val=0.05,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        self.declare_partials(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length",
            "air_flow_length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length"
        ] = np.clip(inputs["air_flow_length"], 0.05, np.inf)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        clipped_air_flow_length = np.clip(inputs["air_flow_length"], 0.05, np.inf)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length",
            "air_flow_length",
        ] = np.where(inputs["air_flow_length"] == clipped_air_flow_length, 1.0, 1e-6)
