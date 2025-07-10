# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SERVICE_BATTERY_OCV

SUBMODEL_BATTERY_OCV_REF_CELL = (
    "fastga_he.submodel.propulsion.battery.open_circuit_voltage.from_reference_cell"
)
SUBMODEL_BATTERY_OCV_MIN_MAX = (
    "fastga_he.submodel.propulsion.battery.open_circuit_voltage.from_min_max"
)

oad.RegisterSubmodel.active_models[SERVICE_BATTERY_OCV] = SUBMODEL_BATTERY_OCV_REF_CELL


@oad.RegisterSubmodel(SERVICE_BATTERY_OCV, SUBMODEL_BATTERY_OCV_REF_CELL)
class PerformancesOpenCircuitVoltage(om.ExplicitComponent):
    """
    Computation of the open circuit voltage of one module cell, takes into account the impact of
    the SOC on the performances. Does not account for temperature just yet, it seems however that
    the dependency on temperature is only visible at very low SOC :cite:`chin:2019`. The model
    put forward by :cite:`baccouche:2017` does not provide satisfactory results but the article
    suggest a simple polynomial fit could provide adequate results, which is what we took. See
    internal_resistance_new.py for how we obtain those values.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Needed for compatibility, unused
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=True,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output("open_circuit_voltage", units="V", val=np.full(number_of_points, 4.1))

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        ocv = (
            -9.65121262e-10 * dod**5.0
            + 1.81419058e-07 * dod**4.0
            - 1.11814100e-05 * dod**3.0
            + 2.26114438e-04 * dod**2.0
            - 8.54619953e-03 * dod
            + 4.12
        )
        outputs["open_circuit_voltage"] = ocv

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        partials["open_circuit_voltage", "state_of_charge"] = -(
            -5.0 * 9.65121262e-10 * dod**4.0
            + 4.0 * 1.81419058e-07 * dod**3.0
            - 3.0 * 1.11814100e-05 * dod**2.0
            + 2.0 * 2.26114438e-04 * dod
            - 8.54619953e-03
        )


@oad.RegisterSubmodel(SERVICE_BATTERY_OCV, SUBMODEL_BATTERY_OCV_REF_CELL)
class PerformancesOpenCircuitVoltageFromMinMax(om.ExplicitComponent):
    """
    Computation of the open circuit voltage of one module cell, takes into account the impact of
    the SOC on the performances. Assumes linear variation of the OCV between a min and max value
    based on the SOC
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Needed for compatibility, unused
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_max_SOC",  # Is the name clear enough ?
            units="V",
            val=np.nan,
            desc="Voltage of the battery cell at a state of charge of 100%",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_min_SOC",
            units="V",
            val=np.nan,
            desc="Voltage of the battery cell at a state of charge of 10%",
        )

        self.add_output("open_circuit_voltage", units="V", val=np.full(number_of_points, 4.1))

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.declare_partials(
            of="open_circuit_voltage",
            wrt="state_of_charge",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="open_circuit_voltage",
            wrt=[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell:voltage_max_SOC",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell:voltage_min_SOC",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        v_max = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_max_SOC"
        ]
        v_min = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_min_SOC"
        ]

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )

        ocv = v_max - (v_max - v_min) * (100.0 - soc) / (100.0 - 10.0)

        outputs["open_circuit_voltage"] = ocv

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        v_max = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_max_SOC"
        ]
        v_min = inputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_min_SOC"
        ]
        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )

        partials["open_circuit_voltage", "state_of_charge"] = np.full(
            number_of_points, (v_max - v_min) / (100.0 - 10.0)
        )
        partials[
            "open_circuit_voltage",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_max_SOC",
        ] = 1.0 - (100.0 - soc) / (100.0 - 10.0)
        partials[
            "open_circuit_voltage",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:voltage_min_SOC",
        ] = (100.0 - soc) / (100.0 - 10.0)
