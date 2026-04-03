# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPEMFCStackBOPDeltaCd(om.ExplicitComponent):
    """The drag created by the flush inlet and the exhaust of the PEMFC stack."""

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

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_drag",
            units="N",
            val=0.0,
            shape=number_of_points,
        )

        self.add_output("delta_Cd", val=1e-5, shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="delta_Cd",
            wrt=[
                "density",
                "true_airspeed",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_drag",
            ],
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
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]

        tms_bop_delta_drag = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag"
        ]

        clipped_tms_bop_delta_drag = np.clip(tms_bop_delta_drag, 0.0, np.inf)

        outputs["delta_Cd"] = clipped_tms_bop_delta_drag / (
            0.5 * density * true_airspeed**2.0 * wing_area
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        density = inputs["density"]
        true_airspeed = inputs["true_airspeed"]
        wing_area = inputs["data:geometry:wing:area"]

        tms_bop_delta_drag = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag"
        ]
        clipped_tms_bop_delta_drag = np.clip(tms_bop_delta_drag, 0.0, np.inf)

        partials["delta_Cd", "density"] = -tms_bop_delta_drag / (
            0.5 * density**2.0 * true_airspeed**2.0 * wing_area
        )

        partials["delta_Cd", "true_airspeed"] = -2.0 * (
            clipped_tms_bop_delta_drag / (0.5 * density * true_airspeed**3.0 * wing_area)
        )

        partials["delta_Cd", "data:geometry:wing:area"] = -clipped_tms_bop_delta_drag / (
            0.5 * density * true_airspeed**2.0 * wing_area**2.0
        )

        partials[
            "delta_Cd",
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag",
        ] = np.where(
            tms_bop_delta_drag == clipped_tms_bop_delta_drag,
            1.0 / (0.5 * density * true_airspeed**2.0 * wing_area),
            1e-6,
        )
