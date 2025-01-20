# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from stdatm import AtmosphereWithPartials

DEFAULT_PRESSURE = 101325.0


class PerformancesOperationPressure(om.ExplicitComponent):
    """
    Computation of the ambient pressure that PEMFC is working based on altitude only applied to the model
    from: `Fuel Cell and Battery Hybrid System Optimization by J. Hoogendoorn:2018`
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(
            name="operation_pressure",
            units="Pa",
            val=np.full(number_of_points, DEFAULT_PRESSURE),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["operation_pressure"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["operation_pressure", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).partial_pressure_altitude
