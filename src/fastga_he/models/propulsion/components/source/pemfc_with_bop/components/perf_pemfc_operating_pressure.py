# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import DEFAULT_PRESSURE


class PerformancesPEMFCStackBOPOperatingPressure(om.ExplicitComponent):
    """
    Operating pressure computation of the PEMFC stack.
    """

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
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air flush_inlet",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_connection = self.options["compressor_connection"]

        if compressor_connection:
            self.add_input(
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":operating_pressure",
                units="Pa",
                val=np.nan,
                desc="Input anode pressure if applicable",
            )
        else:
            self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="operating_pressure",
            units="Pa",
            val=np.full(number_of_points, DEFAULT_PRESSURE),
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        compressor_connection = self.options["compressor_connection"]

        if compressor_connection:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
                val=np.ones(number_of_points),
            )

        else:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_connection = self.options["compressor_connection"]

        if compressor_connection:
            outputs["operating_pressure"] = np.full(
                number_of_points,
                inputs[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":operating_pressure"
                ],
            )
        else:
            outputs["operating_pressure"] = inputs["ambient_pressure"]
