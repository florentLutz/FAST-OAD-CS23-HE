# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import POSSIBLE_POSITION, SUBMODEL_DC_LINE_SIZING_LENGTH

oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_SIZING_LENGTH] = (
    "fastga_he.submodel.propulsion.sizing.dc_line.length.from_position"
)


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_SIZING_LENGTH,
    "fastga_he.submodel.propulsion.sizing.dc_line.length.from_position",
)
class SizingHarnessLength(om.ExplicitComponent):
    """
    Class that computes the length of the DC cable based on the position of its source and target.
    Will be based on input for the geometric position, this module is meant to help reflect the
    scaling of the cable length based on the aircraft dimensions.
    """

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="from_rear_to_front",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the cable harness, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
                val=np.nan,
                desc="Length of the cable as a ratio of the SPAN (not half-span), used when cable "
                "is fully or partly inside the wing.",
            )

        elif position == "from_rear_to_front":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        elif position == "from_rear_to_wing":
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
                val=np.nan,
                desc="Length of the cable as a ratio of the SPAN (not half-span), used when cable "
                "is fully or partly inside the wing.",
            )

        elif position == "from_front_to_wing":
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
                val=np.nan,
                desc="Length of the cable as a ratio of the SPAN (not half-span), used when cable "
                "is fully or partly inside the wing.",
            )

        elif position == "from_rear_to_nose":
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        elif position == "from_front_to_nose":
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        else:
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            units="m",
            val=2.5,
            desc="Length of the harness",
        )
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                inputs[
                    "data:propulsion:he_power_train:DC_cable_harness:"
                    + harness_id
                    + ":length_span_ratio"
                ]
                * inputs["data:geometry:wing:span"]
            )

        elif position == "from_rear_to_front":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                inputs["data:geometry:cabin:length"]
            )

        elif position == "from_rear_to_wing":
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            l_av = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]
            wing_span = inputs["data:geometry:wing:span"]
            span_ratio = inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio"
            ]

            length_in_fuselage = abs(l_av + cabin_length - fa_length)
            length_in_wing = wing_span * span_ratio

            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                length_in_wing + length_in_fuselage
            )

        elif position == "from_front_to_wing":
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            l_av = inputs["data:geometry:fuselage:front_length"]
            wing_span = inputs["data:geometry:wing:span"]
            span_ratio = inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio"
            ]

            length_in_fuselage = abs(l_av - fa_length)
            length_in_wing = wing_span * span_ratio

            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                length_in_wing + length_in_fuselage
            )

        elif position == "from_rear_to_nose":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                inputs["data:geometry:fuselage:front_length"] + inputs["data:geometry:cabin:length"]
            )

        elif position == "from_front_to_nose":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                inputs["data:geometry:fuselage:front_length"]
            )

        else:
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"] = (
                inputs["data:geometry:wing:MAC:at25percent:x"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:wing:span",
            ] = inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
            ] = inputs["data:geometry:wing:span"]

        elif position == "from_rear_to_front":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:cabin:length",
            ] = 1.0

        elif position == "from_rear_to_wing":
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            l_av = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]

            if fa_length > l_av + cabin_length:
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:wing:MAC:at25percent:x",
                ] = 1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:fuselage:front_length",
                ] = -1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:cabin:length",
                ] = -1.0

            else:
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:wing:MAC:at25percent:x",
                ] = -1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:fuselage:front_length",
                ] = 1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:cabin:length",
                ] = 1.0

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:wing:span",
            ] = inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
            ] = inputs["data:geometry:wing:span"]

        elif position == "from_front_to_wing":
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            l_av = inputs["data:geometry:fuselage:front_length"]

            if fa_length > l_av:
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:wing:MAC:at25percent:x",
                ] = 1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:fuselage:front_length",
                ] = -1.0

            else:
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:wing:MAC:at25percent:x",
                ] = -1.0
                partials[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                    "data:geometry:fuselage:front_length",
                ] = 1.0

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:wing:span",
            ] = inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":length_span_ratio",
            ] = inputs["data:geometry:wing:span"]

        elif position == "from_rear_to_nose":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:cabin:length",
            ] = 1.0

        elif position == "from_front_to_nose":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:fuselage:front_length",
            ] = 1.0

        else:
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0
