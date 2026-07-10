# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


# class SlipstreamPEMFCStackBOPDeltaCd(om.ExplicitComponent):
#     """The drag created by the flush flush_inlet and the exhaust of the PEMFC stack."""
#
#     def initialize(self):
#         self.options.declare(
#             "number_of_points", default=1, desc="number of equilibrium to be treated"
#         )
#         self.options.declare(
#             name="pemfc_stack_bop_id",
#             default=None,
#             desc="Identifier of the PEMFC stack",
#             allow_none=False,
#         )
#
#     def setup(self):
#         number_of_points = self.options["number_of_points"]
#         pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
#
#         self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
#         self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
#         self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
#         self.add_input(
#             name="bop_drag",
#             units="N",
#             val=np.array([108.5] * 30 + [159.75] * 30 + [154.4] * 20 + [30.14] * 10),
#         )
#
#         self.add_output("delta_Cd", val=1e-5, shape=number_of_points)
#
#     def setup_partials(self):
#         number_of_points = self.options["number_of_points"]
#         pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
#
#         self.declare_partials(
#             of="delta_Cd",
#             wrt=[
#                 "density",
#                 "true_airspeed",
#                 "bop_drag",
#             ],
#             method="exact",
#             rows=np.arange(number_of_points),
#             cols=np.arange(number_of_points),
#         )
#         self.declare_partials(
#             of="delta_Cd",
#             wrt="data:geometry:wing:area",
#             method="exact",
#             rows=np.arange(number_of_points),
#             cols=np.zeros(number_of_points),
#         )
#
#     def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
#         pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
#
#         density = inputs["density"]
#         true_airspeed = inputs["true_airspeed"]
#         wing_area = inputs["data:geometry:wing:area"]
#
#         tms_bop_delta_drag = inputs["bop_drag"]
#
#         outputs["delta_Cd"] = tms_bop_delta_drag / (0.5 * density * true_airspeed**2.0 * wing_area)
#
#     def compute_partials(self, inputs, partials, discrete_inputs=None):
#         pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
#
#         density = inputs["density"]
#         true_airspeed = inputs["true_airspeed"]
#         wing_area = inputs["data:geometry:wing:area"]
#
#         tms_bop_delta_drag = inputs["bop_drag"]
#
#         partials["delta_Cd", "density"] = -tms_bop_delta_drag / (
#             0.5 * density**2.0 * true_airspeed**2.0 * wing_area
#         )
#
#         partials["delta_Cd", "true_airspeed"] = -2.0 * (
#             tms_bop_delta_drag / (0.5 * density * true_airspeed**3.0 * wing_area)
#         )
#
#         partials["delta_Cd", "data:geometry:wing:area"] = -tms_bop_delta_drag / (
#             0.5 * density * true_airspeed**2.0 * wing_area**2.0
#         )
#
#         partials["delta_Cd", "bop_drag"] = 1.0 / (0.5 * density * true_airspeed**2.0 * wing_area)


class SlipstreamPEMFCStackBOPDeltaCd(om.Group):
    """The drag created by the flush flush_inlet and the exhaust of the PEMFC stack."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_subsystem(
            name="mean_drag",
            subsys=_MeanBOPDrag(
                number_of_points=number_of_points, pemfc_stack_bop_id=pemfc_stack_bop_id
            ),
            promotes=["data:*"],
        )

        self.add_subsystem(
            name="delta_cd",
            subsys=_DeltaCd(number_of_points=number_of_points),
            promotes=["data:*", "density", "true_airspeed", "delta_Cd"],
        )

        self.connect("mean_drag.mean_bop_drag", "delta_cd.mean_bop_drag")


class _MeanBOPDrag(om.ExplicitComponent):
    """The mean drag created by the flush flush_inlet and the exhaust of the PEMFC stack."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_drag",
            units="N",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "mean_bop_drag",
            units="N",
            val=np.array([108.5] * 30 + [159.75] * 30 + [154.4] * 20 + [30.14] * 10),
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="mean_bop_drag",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_drag",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        tms_bop_delta_drag = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag"
        ]

        climb_bop = np.full(30, np.mean(tms_bop_delta_drag[:30]))
        cruise_bop = np.full(30, np.mean(tms_bop_delta_drag[30:60]))
        descent_bop = np.full(20, np.mean(tms_bop_delta_drag[60:80]))
        reserve_bop = np.full(10, np.mean(tms_bop_delta_drag[80:]))

        outputs["mean_bop_drag"] = np.concatenate((climb_bop, cruise_bop, descent_bop, reserve_bop))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        climb_bop_diff = np.full(30, 1 / 30)
        cruise_bop_diff = np.full(30, 1 / 30)
        descent_bop_diff = np.full(20, 1 / 20)
        reserve_bop_diff = np.full(10, 1 / 10)

        partials[
            "mean_bop_drag",
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag",
        ] = np.concatenate((climb_bop_diff, cruise_bop_diff, descent_bop_diff, reserve_bop_diff))


class _DeltaCd(om.ExplicitComponent):
    """The drag created by the flush flush_inlet and the exhaust of the PEMFC stack."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(name="mean_bop_drag", units="N", val=np.nan, shape=number_of_points)

        self.add_output("delta_Cd", val=1e-5, shape=number_of_points, lower=0.0)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="delta_Cd",
            wrt=["density", "true_airspeed", "mean_bop_drag"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cd",
            wrt="data:geometry:wing:area",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]
        tms_bop_delta_drag = inputs["mean_bop_drag"]

        outputs["delta_Cd"] = tms_bop_delta_drag / (0.5 * density * true_airspeed**2.0 * wing_area)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]
        tms_bop_delta_drag = inputs["mean_bop_drag"]

        partials["delta_Cd", "density"] = -tms_bop_delta_drag / (
            0.5 * density**2.0 * true_airspeed**2.0 * wing_area
        )

        partials["delta_Cd", "true_airspeed"] = -2.0 * (
            tms_bop_delta_drag / (0.5 * density * true_airspeed**3.0 * wing_area)
        )

        partials["delta_Cd", "data:geometry:wing:area"] = -tms_bop_delta_drag / (
            0.5 * density * true_airspeed**2.0 * wing_area**2.0
        )

        partials["delta_Cd", "mean_bop_drag"] = 1.0 / (
            0.5 * density * true_airspeed**2.0 * wing_area
        )
