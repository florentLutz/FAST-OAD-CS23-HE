# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import logging

_LOGGER = logging.getLogger(__name__)


class PerformancesModuleCRate(om.ExplicitComponent):
    """
    Computation of the C-rate of each module, assume each module provide an equal amount of
    current and the capacity of one module is equal to the capacity of one cell.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="cell_capacity_ref",
            types=float,
            default=3.35,
            desc="Capacity of the reference cell for the battery construction [A*h]",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":state_of_health",
            val=100.0,
            units="percent",
            desc="State of Health of the battery, i.e. capacity with respect to nominal capacity",
        )

        # Weirdly enough, this will be a output of a performance module, it does not really make
        # sense to create a component in sizing that just outputs this value based on an option
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity",
            val=self.options["cell_capacity_ref"],
            units="A*h",
            desc="Capacity of the cell used for the assembly of the battery pack",
        )
        self.add_output("c_rate", units="h**-1", val=np.full(number_of_points, 1.0))

        self.declare_partials(
            of="c_rate",
            wrt="current_one_module",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="c_rate",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":state_of_health",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":state_of_health",
            method="exact",
            val=self.options["cell_capacity_ref"] / 100.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        current = np.where(
            np.abs(inputs["current_one_module"]) < 1e-2, 0.0, inputs["current_one_module"]
        )
        soh = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":state_of_health"
        ]

        outputs["c_rate"] = current / (self.options["cell_capacity_ref"] * soh / 100.0)
        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity"
        ] = self.options["cell_capacity_ref"] * soh / 100.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        soh = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":state_of_health"
        ]
        current = np.where(
            np.abs(inputs["current_one_module"]) < 1e-2, 0.0, inputs["current_one_module"]
        )

        partials_current = np.where(
            np.abs(inputs["current_one_module"]) < 1e-2,
            1e-6,
            1.0 / (self.options["cell_capacity_ref"] * soh / 100.0),
        )
        partials["c_rate", "current_one_module"] = partials_current
        partials[
            "c_rate",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":state_of_health",
        ] = -current / (self.options["cell_capacity_ref"] * soh**2.0 / 100.0)
