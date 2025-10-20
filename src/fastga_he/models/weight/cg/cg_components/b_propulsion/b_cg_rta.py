"""
Estimation of propulsion center of gravity
"""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel
from openmdao.core.explicitcomponent import ExplicitComponent
from rta.models.weight.cg.constants import SERVICE_PROPULSION_CG


@RegisterSubmodel(SERVICE_PROPULSION_CG, "fastga_he.submodel.weight.cg.nacelle.rta")
class ComputeNacelleCGRTA(om.Group):
    """Nacelle center of gravity estimation as a function of wing position"""

    def setup(self):
        self.add_subsystem(name="y_nacelle", subsys=_YNacelle(), promotes=["data:*"])
        self.add_subsystem(name="delta_x_nacelle", subsys=_DeltaXNacelle(), promotes=["data:*"])
        self.add_subsystem(name="nacelle_cg", subsys=_NacelleCG(), promotes=["data:*"])

        self.connect("y_nacelle.y_nacelle", "delta_x_nacelle.y_nacelle")
        self.connect("y_nacelle.y_nacelle", "nacelle_cg.y_nacelle")
        self.connect("delta_x_nacelle.delta_x_nacelle", "nacelle_cg.delta_x_nacelle")


class _YNacelle(om.ExplicitComponent):
    """The position of the nacelle along the y-direction (wing span)"""

    def setup(self):
        self.add_input("data:geometry:propulsion:engine:y_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("y_nacelle", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y_nacelle"] = (
            inputs["data:geometry:propulsion:engine:y_ratio"]
            * inputs["data:geometry:wing:span"]
            / 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["y_nacelle", "data:geometry:propulsion:engine:y_ratio"] = (
            inputs["data:geometry:wing:span"] / 2.0
        )

        partials["y_nacelle", "data:geometry:wing:span"] = (
            inputs["data:geometry:propulsion:engine:y_ratio"] / 2.0
        )


class _DeltaXNacelle(ExplicitComponent):
    """Propulsion center of gravity estimation as a function of wing position"""

    def setup(self):
        self.add_input("y_nacelle", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")

        self.add_output("delta_x_nacelle", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y_nacelle = inputs["y_nacelle"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        y3_wing = inputs["data:geometry:wing:kink:y"]

        outputs["delta_x_nacelle"] = 0.05 * (
            l3_wing + (l2_wing - l3_wing) * (y3_wing - y_nacelle) / (y3_wing - y2_wing)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y_nacelle = inputs["y_nacelle"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        y3_wing = inputs["data:geometry:wing:kink:y"]

        partials["delta_x_nacelle", "data:geometry:wing:root:chord"] = (
            0.05 * (y3_wing - y_nacelle) / (y3_wing - y2_wing)
        )

        partials["delta_x_nacelle", "data:geometry:wing:kink:chord"] = 0.05 * (
            1.0 - (y3_wing - y_nacelle) / (y3_wing - y2_wing)
        )

        partials["delta_x_nacelle", "y_nacelle"] = -0.05 * (l2_wing - l3_wing) / (y3_wing - y2_wing)

        partials["delta_x_nacelle", "data:geometry:wing:kink:y"] = (
            0.05 * (l2_wing - l3_wing) * (y_nacelle - y2_wing) / (y3_wing - y2_wing) ** 2.0
        )

        partials["delta_x_nacelle", "data:geometry:wing:root:y"] = (
            0.05 * (l2_wing - l3_wing) * (y3_wing - y_nacelle) / (y3_wing - y2_wing) ** 2.0
        )


class _NacelleCG(ExplicitComponent):
    """Computation for Nacelle absolute CG position"""

    def setup(self):
        self.add_input("y_nacelle", val=np.nan, units="m")
        self.add_input("delta_x_nacelle", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:airframe:nacelle:CG:x", units="m")

    def setup_partials(self):
        self.declare_partials("data:weight:airframe:nacelle:CG:x", "*", method="exact")
        self.declare_partials(
            "data:weight:airframe:nacelle:CG:x", "data:geometry:wing:MAC:at25percent:x", val=1.025
        )
        self.declare_partials(
            "data:weight:airframe:nacelle:CG:x", "data:geometry:wing:MAC:length", val=-0.25625
        )
        self.declare_partials(
            "data:weight:airframe:nacelle:CG:x",
            ["data:geometry:wing:MAC:leading_edge:x:local", "delta_x_nacelle"],
            val=-1.025,
        )
        self.declare_partials(
            "data:weight:airframe:nacelle:CG:x",
            "data:geometry:propulsion:nacelle:length",
            val=-0.205,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y_nacelle = inputs["y_nacelle"]
        delta_x_nacelle = inputs["delta_x_nacelle"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        x3_wing = inputs["data:geometry:wing:kink:leading_edge:x:local"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

        x_nacelle_cg = (
            x3_wing * (y_nacelle - y2_wing) / (y3_wing - y2_wing)
            - delta_x_nacelle
            - 0.2 * nac_length
        )

        outputs["data:weight:airframe:nacelle:CG:x"] = 1.025 * (
            fa_length - 0.25 * l0_wing - (x0_wing - x_nacelle_cg)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y_nacelle = inputs["y_nacelle"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        x3_wing = inputs["data:geometry:wing:kink:leading_edge:x:local"]
        y3_wing = inputs["data:geometry:wing:kink:y"]

        partials[
            "data:weight:airframe:nacelle:CG:x", "data:geometry:wing:kink:leading_edge:x:local"
        ] = 1.025 * (y_nacelle - y2_wing) / (y3_wing - y2_wing)

        partials["data:weight:airframe:nacelle:CG:x", "y_nacelle"] = 1.025 / (y3_wing - y2_wing)

        partials["data:weight:airframe:nacelle:CG:x", "data:geometry:wing:root:y"] = (
            1.025 * x3_wing * (y_nacelle - y3_wing) / (y3_wing - y2_wing) ** 2.0
        )

        partials["data:weight:airframe:nacelle:CG:x", "data:geometry:wing:kink:y"] = (
            1.025 * x3_wing * (y2_wing - y_nacelle) / (y3_wing - y2_wing) ** 2.0
        )
