# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class ComputePayloadRangeInnerSampling(om.ExplicitComponent):
    """
    Computation of the location of the points inside the domain delimited by the payload range
    that will be used to draw the emission factor map.
    """

    def initialize(self):

        self.options.declare(
            name="number_of_sample",
            types=int,
            default=12,
            desc="Number of sample inside the payload range envelope",
        )

    def setup(self):

        number_of_sample = self.options["number_of_sample"]

        self.add_input("data:mission:payload_range:range", val=np.nan, units="NM", shape=4)
        self.add_input("data:mission:payload_range:payload", val=np.nan, units="kg", shape=4)

        self.add_output(
            "data:mission:inner_payload_range:range", val=1.0, units="NM", shape=number_of_sample
        )
        self.add_output(
            "data:mission:inner_payload_range:payload", val=1.0, units="kg", shape=number_of_sample
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_sample = self.options["number_of_sample"]

        outer_range_array = inputs["data:mission:payload_range:range"]
        outer_payload_array = inputs["data:mission:payload_range:payload"]

        # First find the number of payload and the number of range we will compute
        starting_point = int(np.sqrt(number_of_sample))
        for factor in np.arange(starting_point, 0, -1):
            if number_of_sample % factor == 0:
                number_of_ranges = factor
                break

        number_of_payload = int(number_of_sample / number_of_ranges)

        payload_array = (
            (np.linspace(0, number_of_payload - 1, number_of_payload) + 0.5)
            * max(outer_payload_array)
            / number_of_payload
        )

        range_array = np.array([])

        for payload in payload_array:
            max_range_that_payload = np.interp(
                payload, np.flip(outer_payload_array[1:]), np.flip(outer_range_array[1:])
            )

            range_array_that_payload = (
                (np.linspace(0, number_of_ranges - 1, number_of_ranges) + 0.5)
                * max_range_that_payload
                / number_of_ranges
            )
            range_array = np.concatenate((range_array, range_array_that_payload))

        outputs["data:mission:inner_payload_range:range"] = range_array
        outputs["data:mission:inner_payload_range:payload"] = np.repeat(
            payload_array, number_of_ranges
        )
