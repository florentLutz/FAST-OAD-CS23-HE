# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.exceptions import ControlParameterInconsistentShapeError

from ..constants import SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE

SUBMODEL_DC_LINE_TEMPERATURE_CONSTANT = (
    "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.constant"
)


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
    SUBMODEL_DC_LINE_TEMPERATURE_CONSTANT,
)
class PerformancesTemperatureConstant(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        # These need to be there for compatibility, even if they aren't use (submodels and all of
        # that)
        self.add_input(
            "exterior_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature outside of the cable",
            shape=number_of_points,
        )
        self.add_input("time_step", shape=number_of_points, units="s", val=np.nan)

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable_temperature_mission",
            val=np.nan,
            units="degK",
            desc="Performances of the cable will be computed assuming it has a constant "
            "temperature of that value, other submodels exist for steady state and dynamic "
            "temperature profile",
            shape_by_conn=True,
        )

        self.add_output(
            "cable_temperature",
            val=np.full(number_of_points, 288.15),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
            lower=1.0,
        )

        self.declare_partials(
            of="cable_temperature",
            wrt="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable_temperature_mission",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        t_cable_mission = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable_temperature_mission"
        ]

        if len(t_cable_mission) == 1:
            outputs["cable_temperature"] = np.full(number_of_points, t_cable_mission)

        elif len(t_cable_mission) == number_of_points:
            outputs["cable_temperature"] = t_cable_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable_temperature_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        t_cable_mission = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable_temperature_mission"
        ]

        if len(t_cable_mission) == 1:
            partials[
                "cable_temperature",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable_temperature_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "cable_temperature",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable_temperature_mission",
            ] = np.eye(number_of_points)
