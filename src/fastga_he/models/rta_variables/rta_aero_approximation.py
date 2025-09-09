# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem("fastga_he.rta_variables.aero_approx", domain=ModelDomain.GEOMETRY)
class AeroApproximation(om.Group):
    """Computation of the CL_ref and CD_ind based on an elliptic distribution assumption"""

    def setup(self):
        self.add_subsystem(
            "length_vector",
            _LengthVector(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "l_y",
            _Ly(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "vector_product",
            _VectorProduct(),
        )
        self.add_subsystem(
            "cl_ref",
            _Cl_Ref(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "induced_drag_ratio",
            _Induced_Drag_Coefficient(),
            promotes=["data:*"],
        )

        self.connect("length_vector.half_wing_coordinate", "l_y.half_wing_coordinate")
        self.connect("length_vector.chord_vector", "vector_product.chord_vector")
        self.connect("l_y.l_y", "vector_product.l_y")
        self.connect("vector_product.vector_product", "cl_ref.vector_product")


class _LengthVector(om.ExplicitComponent):
    """Computation to construct half wing coordinate and chord vector array ."""

    def setup(self):
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("half_wing_coordinate", shape=100)
        self.add_output("chord_vector", shape=100)

    def setup_partials(self):
        self.declare_partials(
            "half_wing_coordinate",
            "data:geometry:wing:b_50",
            val=np.linspace(0.0, 0.5, 100),
            method="exact",
            rows=np.arange(100),
            cols=np.zeros(100),
        )
        self.declare_partials(
            "chord_vector",
            "data:geometry:wing:tip:chord",
            val=np.linspace(0.0, 1.0, 100),
            method="exact",
            rows=np.arange(100),
            cols=np.zeros(100),
        )
        self.declare_partials(
            "chord_vector",
            "data:geometry:wing:root:chord",
            val=np.linspace(1.0, 0.0, 100),
            method="exact",
            rows=np.arange(100),
            cols=np.zeros(100),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["half_wing_coordinate"] = np.linspace(
            0, inputs["data:geometry:wing:b_50"].item() / 2.0, 100
        )

        outputs["chord_vector"] = np.linspace(
            inputs["data:geometry:wing:root:chord"].item(),
            inputs["data:geometry:wing:tip:chord"].item(),
            100,
        )


class _Ly(om.ExplicitComponent):
    """Computation of the cl_ref based on an elliptic distribution assumption"""

    def setup(self):
        self.add_input("half_wing_coordinate", val=np.nan, shape=100)
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")

        self.add_output("l_y", shape=100)

    def setup_partials(self):
        self.declare_partials(
            "l_y", "half_wing_coordinate", method="exact", rows=np.arange(100), cols=np.arange(100)
        )
        self.declare_partials(
            "l_y",
            "data:geometry:wing:b_50",
            method="exact",
            rows=np.arange(100),
            cols=np.zeros(100),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["l_y"] = (
            np.ones(100)
            - (2.0 * inputs["half_wing_coordinate"] / inputs["data:geometry:wing:b_50"]) ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["l_y", "half_wing_coordinate"] = (
            -8.0 * inputs["half_wing_coordinate"] / inputs["data:geometry:wing:b_50"] ** 2.0
        )

        partials["l_y", "data:geometry:wing:b_50"] = (
            8.0 * inputs["half_wing_coordinate"] ** 2.0 / inputs["data:geometry:wing:b_50"] ** 3.0
        )


class _VectorProduct(om.ExplicitComponent):
    """Computation of the vector product in cl_ref based on an elliptic distribution assumption"""

    def setup(self):
        self.add_input("l_y", val=np.nan, shape=100)
        self.add_input("chord_vector", val=np.nan, shape=100)

        self.add_output("vector_product", shape=100)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact", rows=np.arange(100), cols=np.arange(100))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["vector_product"] = inputs["l_y"] * inputs["chord_vector"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["vector_product", "chord_vector"] = inputs["l_y"]

        partials["vector_product", "l_y"] = inputs["chord_vector"]


class _Cl_Ref(om.ExplicitComponent):
    """Computation of the cl_ref based on an elliptic distribution assumption"""

    def setup(self):
        self.add_input("vector_product", val=np.nan, shape=100)
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:aerodynamics:wing:low_speed:CL_ref", val=0.67888)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_w = inputs["data:geometry:wing:area"]
        product = inputs["vector_product"]
        b = inputs["data:geometry:wing:b_50"]

        outputs["data:aerodynamics:wing:low_speed:CL_ref"] = np.trapz(product, dx=b / 198.0) / (
            0.5 * s_w
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_w = inputs["data:geometry:wing:area"]
        product = inputs["vector_product"]
        b = inputs["data:geometry:wing:b_50"]

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:area"] = -np.trapz(
            product, dx=b / 198.0
        ) / (0.5 * s_w**2.0)

        partials["data:aerodynamics:wing:low_speed:CL_ref", "vector_product"] = (
            np.array([b / 198.0] + [b / 99.0] * 98 + [b / 198.0]) / s_w
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:b_50"] = np.trapz(
            product, dx=1.0 / 198.0
        ) / (0.5 * s_w)


class _Induced_Drag_Coefficient(om.ExplicitComponent):
    """Computation of the induced drag coefficient in cruise"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:oswald_coefficient", val=np.nan)

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=0.08234
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = (
            np.pi
            * inputs["data:geometry:horizontal_tail:aspect_ratio"]
            * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"]
        ) ** -1.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = -(
            (
                np.pi
                * inputs["data:geometry:horizontal_tail:aspect_ratio"] ** 2.0
                * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"]
            )
            ** -1.0
        )

        partials[
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
            "data:aerodynamics:aircraft:cruise:oswald_coefficient",
        ] = -(
            (
                np.pi
                * inputs["data:geometry:horizontal_tail:aspect_ratio"]
                * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"] ** 2.0
            )
            ** -1.0
        )
