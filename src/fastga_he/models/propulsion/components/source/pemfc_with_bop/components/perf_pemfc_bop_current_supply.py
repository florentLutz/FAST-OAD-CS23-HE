# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import (
    FARADAY_CONSTANT,
    H2_MOL_PER_KG,
    NUMBER_OF_ELETRONS_FROM_H2,
    DEFAULT_HYDROGEN_CONSUMPTION,
)


class PerformancesPEMFCStackBOPCurrentSupply(om.ExplicitComponent):
    """
    Computation of the current supply of the PEMFC, including the power required
    of the pump and the compressor.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            units="W",
            val=0.0,
            shape=number_of_points,
        )
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output(
            "pemfc_dc_current",
            units="A",
            val=300.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="pemfc_dc_current",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        bop_power_required = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ]
        dc_current_out = inputs["dc_current_out"]
        voltage_out = inputs["voltage_out"]

        outputs["pemfc_dc_current"] = dc_current_out + bop_power_required / voltage_out

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        bop_power_required = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ]
        voltage_out = inputs["voltage_out"]

        partials["pemfc_dc_current", "dc_current_out"] = np.ones(number_of_points)

        partials[
            "pemfc_dc_current",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
        ] = 1.0 / voltage_out

        partials["pemfc_dc_current", "voltage_out"] = -bop_power_required / voltage_out**2.0
