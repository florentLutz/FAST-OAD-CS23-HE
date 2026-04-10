# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_POWER, MAX_DEFAULT_CURRENT, DEFAULT_AIR_CONSUMPTION


class PerformancesPEMFCStackBOPMaximum(om.ExplicitComponent):
    """
    Computation that identifies the maximum power and current output from the PEMFC stack.
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
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))
        self.add_input("thermal_power", units="kW", val=np.full(number_of_points, np.nan))
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("air_consumption", units="kg/s", val=np.full(number_of_points, np.nan))
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
            units="kW",
            val=MAX_DEFAULT_POWER,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
            units="kW",
            val=MAX_DEFAULT_POWER,
            desc="Maximum total thermal power of the PEMFC stack has to provide during the mission",
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating",
            units="kW",
            val=0.0,
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":current_max",
            units="A",
            val=MAX_DEFAULT_CURRENT,
            desc="Maximum current the PEMFC stack has to provide during mission",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            units="kg/s",
            val=DEFAULT_AIR_CONSUMPTION,
            desc="Maximum air consumption of the PEMFC during mission",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":power_max",
            wrt="power_out",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
            wrt="thermal_power",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":current_max",
            wrt="dc_current_out",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            wrt="air_consumption",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max"
        ] = np.max(inputs["power_out"])
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max"
        ] = np.max(inputs["thermal_power"])
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating"
        ] = np.max(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_power_required"
            ]
        )
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":current_max"
        ] = np.max(inputs["dc_current_out"])
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ] = np.max(inputs["air_consumption"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
            "power_out",
        ] = np.where(inputs["power_out"] == np.max(inputs["power_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
            "thermal_power",
        ] = np.where(inputs["thermal_power"] == np.max(inputs["thermal_power"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":current_max",
            "dc_current_out",
        ] = np.where(inputs["dc_current_out"] == np.max(inputs["dc_current_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            "air_consumption",
        ] = np.where(inputs["air_consumption"] == np.max(inputs["air_consumption"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_rating",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
        ] = np.where(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_power_required"
            ]
            == np.max(
                inputs[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":bop_power_required"
                ]
            ),
            1.0,
            0.0,
        )
