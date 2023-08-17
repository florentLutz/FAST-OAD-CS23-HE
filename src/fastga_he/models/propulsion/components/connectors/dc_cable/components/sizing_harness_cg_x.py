# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingHarnessCGX(om.ExplicitComponent):
    """
    Class that computes the CG of the DC cable based on the position of its source and target.
    Will be based on simple geometric ratios, no consideration of volume will be implemented for
    now.
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

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "from_rear_to_front":

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        elif position == "from_rear_to_wing":

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "from_front_to_wing":

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "from_rear_to_nose":

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        elif position == "from_front_to_nose":

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        else:

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the DC cable center of gravity",
        )
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            outputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"
            ] = inputs["data:geometry:wing:MAC:at25percent:x"]

        elif position == "from_rear_to_front":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                inputs["data:geometry:fuselage:front_length"]
                + 0.5 * inputs["data:geometry:cabin:length"]
            )

        elif position == "from_rear_to_wing":

            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            cabin_length = inputs["data:geometry:cabin:length"]
            lav = inputs["data:geometry:fuselage:front_length"]

            rear_to_wing = cabin_length + lav - fa_length
            # We put a heavier weight on the location of the wing to emulate the fact that if the
            # source is in the rear and the target in the wing, some cable length will be need to
            # go from fuselage center line to wherever on the wing the target is
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                fa_length + 1.0 / 3.0 * rear_to_wing
            )

        elif position == "from_front_to_wing":

            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
            lav = inputs["data:geometry:fuselage:front_length"]

            front_to_wing = lav - fa_length
            # Same reasoning as the case before
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                fa_length + 1.0 / 3.0 * front_to_wing
            )

        elif position == "from_rear_to_nose":

            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                inputs["data:geometry:fuselage:front_length"] + inputs["data:geometry:cabin:length"]
            ) / 2.0

        elif position == "from_front_to_nose":

            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                inputs["data:geometry:fuselage:front_length"] / 2.0
            )

        else:

            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x"] = (
                inputs["data:geometry:wing:MAC:at25percent:x"] / 2.0
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        elif position == "from_rear_to_front":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 0.5

        elif position == "from_rear_to_wing":

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = (
                2.0 / 3.0
            )
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = (
                1.0 / 3.0
            )
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = (
                1.0 / 3.0
            )

        elif position == "from_front_to_wing":

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = (
                2.0 / 3.0
            )
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = (
                1.0 / 3.0
            )

        elif position == "from_rear_to_nose":

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 0.5
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 0.5

        elif position == "from_front_to_nose":

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 0.5

        else:

            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 0.5
