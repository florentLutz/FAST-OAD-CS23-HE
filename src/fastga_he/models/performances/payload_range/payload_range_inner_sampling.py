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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.number_of_ranges = 0
        self.number_of_payload = 0

    def initialize(self):

        self.options.declare(
            name="number_of_sample",
            types=int,
            default=12,
            desc="Number of sample inside the payload range envelope. Some points are added by default on the envelope",
        )

    def setup(self):

        number_of_sample = self.options["number_of_sample"]

        starting_point = int(np.sqrt(number_of_sample))
        for factor in np.arange(starting_point, 0, -1):
            if number_of_sample % factor == 0:
                self.number_of_ranges = factor
                break

        self.number_of_payload = int(number_of_sample / self.number_of_ranges)

        self.add_input("data:mission:payload_range:range", val=np.nan, units="NM", shape=4)
        self.add_input("data:mission:payload_range:payload", val=np.nan, units="kg", shape=4)

        self.add_output(
            "data:mission:inner_payload_range:range",
            val=1.0,
            units="NM",
            shape=(self.number_of_payload + 3) * (self.number_of_ranges + 2),
        )
        self.add_output(
            "data:mission:inner_payload_range:payload",
            val=1.0,
            units="kg",
            shape=(self.number_of_payload + 3) * (self.number_of_ranges + 2),
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outer_range_array = inputs["data:mission:payload_range:range"]
        outer_payload_array = inputs["data:mission:payload_range:payload"]

        max_payload = max(outer_payload_array)

        min_payload = max_payload / 15.0
        min_range = max(outer_range_array) / 15.0

        payload_array = (
            (np.linspace(0, self.number_of_payload - 1, self.number_of_payload) + 0.5)
            * max(outer_payload_array)
            / self.number_of_payload
        )
        payload_array = np.concatenate(
            (
                np.array([min_payload]),
                payload_array,
                np.array([max_payload]),
                np.array([outer_payload_array[2]]),
            )
        )
        payload_array.sort()

        range_array = np.array([])
        payload_mesh = np.array([])

        for payload in payload_array:
            max_range_that_payload = np.interp(
                payload, np.flip(outer_payload_array[1:]), np.flip(outer_range_array[1:])
            )

            range_array_that_payload = (
                (np.linspace(0, self.number_of_ranges - 1, self.number_of_ranges) + 0.5)
                * max_range_that_payload
                / self.number_of_ranges
            )

            range_array = np.concatenate(
                (
                    range_array,
                    np.array([min_range]),
                    range_array_that_payload,
                    np.array([max_range_that_payload]),
                )
            )
            payload_mesh = np.concatenate(
                (
                    payload_mesh,
                    np.array([payload]),
                    np.full_like(range_array_that_payload, payload),
                    np.array([payload]),
                )
            )

        outputs["data:mission:inner_payload_range:range"] = range_array
        outputs["data:mission:inner_payload_range:payload"] = payload_mesh
