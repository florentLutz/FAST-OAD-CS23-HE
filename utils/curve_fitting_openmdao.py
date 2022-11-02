# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


def curve_fit_openmdao(speed_array, torque_array, efficiency_target):

    size = len(speed_array)

    ivc = om.IndepVarComp()
    ivc.add_output("torque", torque_array, shape=size, units="N*m")
    ivc.add_output("speed", speed_array, shape=size, units="rad/s")
    ivc.add_output("target_efficiency", efficiency_target, shape=size)

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem("inputs", ivc, promotes_outputs=["*"])
    model.add_subsystem("power_losses", PowerLossPolito(number_of_points=size), promotes=["*"])
    model.add_subsystem("efficiency", Efficiency(number_of_points=size), promotes=["*"])
    model.add_subsystem(
        "difference_to_target", DifferenceToTarget(number_of_points=size), promotes=["*"]
    )

    model.nonlinear_solver = om.NonlinearBlockGS()
    model.nonlinear_solver.options["iprint"] = 2
    model.nonlinear_solver.options["maxiter"] = 100
    model.nonlinear_solver.options["rtol"] = 1e-8
    model.linear_solver = om.LinearBlockGS()

    problem.driver = om.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"
    problem.driver.options["maxiter"] = 100
    problem.driver.options["tol"] = 1e-8

    problem.model.add_design_var(
        name="c_0",
        lower=0.0,
    )
    problem.model.add_design_var(
        name="c_02",
        lower=0.0,
    )
    problem.model.add_design_var(
        name="c_20",
        lower=0.0,
    )
    problem.model.add_design_var(
        name="c_22",
        lower=0.0,
    )
    problem.model.add_design_var(
        name="c_30",
        lower=0.0,
    )

    problem.model.add_objective(name="difference")

    problem.model.approx_totals()
    problem.setup()
    problem.run_driver()

    return problem["c_0"]


class PowerLossPolito(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", default=1)

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("torque", units="N*m", val=np.full(number_of_points, np.nan))
        self.add_input("speed", units="rad/s", val=np.full(number_of_points, np.nan))

        self.add_input("c_0", val=0.01)
        self.add_input("c_20", val=0.01)
        self.add_input("c_02", val=0.01)
        self.add_input("c_22", val=0.01)
        self.add_input("c_30", val=0.01)

        self.add_output("power_losses", val=np.full(number_of_points, 0.0), units="W")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        speed = inputs["speed"]
        torque = inputs["torque"]

        power_losses = (
            inputs["c_0"]
            + inputs["c_20"] * speed ** 2.0
            + inputs["c_02"] * torque ** 2.0
            + inputs["c_22"] * (speed * torque) ** 2.0
            + inputs["c_30"] * speed ** 3.0
        )

        outputs["power_losses"] = power_losses

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        speed = inputs["speed"]
        torque = inputs["torque"]

        partials["power_losses", "c_0"] = np.full(number_of_points, 1.0)
        partials["power_losses", "c_20"] = speed ** 2.0
        partials["power_losses", "c_02"] = torque ** 2.0
        partials["power_losses", "c_22"] = (speed * torque) ** 2.0
        partials["power_losses", "c_30"] = speed ** 3.0

        partials["power_losses", "speed"] = np.diag(
            +2.0 * inputs["c_20"] * speed
            + 2.0 * inputs["c_22"] * speed * torque ** 2.0
            + 3.0 * inputs["c_30"] * speed ** 2.0
        )
        partials["power_losses", "torque"] = np.diag(
            +2.0 * inputs["c_02"] * torque + 2.0 * inputs["c_22"] * torque * speed ** 2.0
        )


class PowerLoss(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", default=1)

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("torque", units="N*m", val=np.full(number_of_points, np.nan))
        self.add_input("speed", units="rad/s", val=np.full(number_of_points, np.nan))

        self.add_input("alpha", val=0.01)
        self.add_input("beta", val=0.01)

        self.add_output("power_losses", val=np.full(number_of_points, 0.0), units="W")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power_losses = (
            inputs["alpha"] * inputs["torque"] ** 2.0 + inputs["beta"] * inputs["speed"] ** 1.5
        )

        outputs["power_losses"] = power_losses

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["power_losses", "alpha"] = inputs["torque"] ** 2.0
        partials["power_losses", "beta"] = inputs["speed"] ** 1.5
        partials["power_losses", "torque"] = np.diag(2.0 * inputs["alpha"] * inputs["torque"])
        partials["power_losses", "speed"] = np.diag(1.5 * inputs["beta"] * inputs["speed"] ** 0.5)


class Efficiency(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", default=1)

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("torque", units="N*m", val=np.full(number_of_points, np.nan))
        self.add_input("speed", units="rad/s", val=np.full(number_of_points, np.nan))
        self.add_input("power_losses", val=np.full(number_of_points, np.nan), units="W")

        self.add_output(
            "computed_efficiency",
            val=np.full(number_of_points, 1.0),
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        computed_efficiency = (inputs["torque"] * inputs["speed"]) / (
            inputs["torque"] * inputs["speed"] + inputs["power_losses"]
        )

        outputs["computed_efficiency"] = computed_efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["computed_efficiency", "torque"] = np.diag(
            inputs["speed"]
            * inputs["power_losses"]
            / (inputs["torque"] * inputs["speed"] + inputs["power_losses"]) ** 2.0
        )
        partials["computed_efficiency", "speed"] = np.diag(
            inputs["torque"]
            * inputs["power_losses"]
            / (inputs["torque"] * inputs["speed"] + inputs["power_losses"]) ** 2.0
        )
        partials["computed_efficiency", "power_losses"] = -np.diag(
            inputs["torque"]
            * inputs["speed"]
            / (inputs["torque"] * inputs["speed"] + inputs["power_losses"]) ** 2.0
        )


class DifferenceToTarget(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", default=1)

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("computed_efficiency", val=np.full(number_of_points, np.nan))
        self.add_input("target_efficiency", val=np.full(number_of_points, np.nan))

        self.add_output("difference", val=1.0)

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["difference"] = np.sqrt(
            np.mean((inputs["computed_efficiency"] - inputs["target_efficiency"]) ** 2.0)
        )
