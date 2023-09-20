# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import time

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from tests.testing_utilities import run_system

NB_POINTS = 90
NB_LOOPS_TEST = 15


class PartialDerivationOldWay(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", default=1, desc="number of points")

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("input_1", shape=number_of_points, val=np.nan)
        self.add_input("input_2", shape=number_of_points, val=np.nan)
        self.add_input("input_3", shape=number_of_points, val=np.nan)

        self.add_output("output_1", shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["output_1"] = (
            np.square(1.0 / inputs["input_1"]) * inputs["input_2"] + 3.0 * inputs["input_3"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["output_1", "input_1"] = np.diag(
            -2.0 * inputs["input_2"] / inputs["input_1"] ** 3.0
        )
        partials["output_1", "input_2"] = np.diag(np.square(1.0 / inputs["input_1"]))
        partials["output_1", "input_3"] = np.eye(number_of_points) * 3.0


class PartialDerivationNewWay(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", default=1, desc="number of points")

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("input_1", shape=number_of_points, val=np.nan)
        self.add_input("input_2", shape=number_of_points, val=np.nan)
        self.add_input("input_3", shape=number_of_points, val=np.nan)

        self.add_output("output_1", shape=number_of_points)

        idx = np.arange(number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact", rows=idx, cols=idx)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["output_1"] = (
            np.square(1.0 / inputs["input_1"]) * inputs["input_2"] + 3.0 * inputs["input_3"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["output_1", "input_1"] = -2.0 * inputs["input_2"] / inputs["input_1"] ** 3.0

        partials["output_1", "input_2"] = np.square(1.0 / inputs["input_1"])
        partials["output_1", "input_3"] = np.full(number_of_points, 3.0)


if __name__ == "__main__":

    init_time_old_way = time.time()

    for i in range(NB_LOOPS_TEST):

        ivc = om.IndepVarComp()
        ivc.add_output("input_1", val=1.0 + 3.0 * np.random.random(NB_POINTS))
        ivc.add_output("input_2", val=1.0e-3 + np.random.random(NB_POINTS))
        ivc.add_output("input_3", val=1.0 - 2.0 * np.random.random(NB_POINTS))

        problem = run_system(PartialDerivationOldWay(number_of_points=NB_POINTS), ivc)

        partials_verification = problem.check_partials(out_stream=None)

        assert_check_partials(partials_verification, atol=1e-5, rtol=1e-5)

    end_time_old_way = time.time()

    print(
        "Timer for old way of defining partials", (end_time_old_way - init_time_old_way) / NB_POINTS
    )

    init_time_new_way = time.time()

    for i in range(NB_LOOPS_TEST):
        ivc = om.IndepVarComp()
        ivc.add_output("input_1", val=1.0 + 3.0 * np.random.random(NB_POINTS))
        ivc.add_output("input_2", val=1.0e-3 + np.random.random(NB_POINTS))
        ivc.add_output("input_3", val=1.0 - 2.0 * np.random.random(NB_POINTS))

        problem = run_system(PartialDerivationNewWay(number_of_points=NB_POINTS), ivc)

        partials_verification = problem.check_partials(out_stream=None)

        assert_check_partials(partials_verification, atol=1e-5, rtol=1e-5)

    end_time_new_way = time.time()

    print(
        "Timer for new way of defining partials", (end_time_new_way - init_time_new_way) / NB_POINTS
    )
