# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesInternalResistance(om.ExplicitComponent):
    """
    Computation of the internal resistance of the battery. :cite:`chen:2021` suggest that
    temperature and SOC dependency can be decoupled which is what we are doing here. For the
    temperature part we will take the same shape as :cite:`chen:2021`. See
    methodology/internal_resistance_temperature.py to see how it was obtained. For the dependency
    part a simple polynomial fit was used, see methodology/internal_resistance_new.py. Data are
    from https://lygte-info.dk/review/batteries2012/Samsung%20INR18650-35E%203500mAh%20%28Pink%29
    %202%20UK.html
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

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))
        self.add_input("cell_temperature", units="degK", val=np.full(number_of_points, np.nan))

        self.add_input(
            name="settings:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":reference_temperature",
            val=293.15,
            units="degK",
        )

        self.add_output("internal_resistance", units="ohm", val=np.full(number_of_points, 1e-3))

        self.declare_partials(
            of="internal_resistance",
            wrt=["state_of_charge", "cell_temperature"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="internal_resistance",
            wrt="settings:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":reference_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        temperature_ref = inputs[
            "settings:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":reference_temperature"
        ]
        cell_temperature = inputs["cell_temperature"]
        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        internal_resistance = (
            2.62771800e-11 * dod**5.0
            - 1.48987233e-08 * dod**4.0
            + 2.03615618e-06 * dod**3.0
            - 1.06451730e-04 * dod**2.0
            + 2.13818712e-03 * dod
            + 3.90444549e-02
        ) * np.exp(
            (46.39 * (temperature_ref - cell_temperature))
            / ((cell_temperature - 254.33) * (temperature_ref - 254.33))
        )

        outputs["internal_resistance"] = internal_resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        temperature_ref = inputs[
            "settings:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":reference_temperature"
        ]
        cell_temperature = inputs["cell_temperature"]
        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        temperature_effect = np.exp(
            (46.39 * (temperature_ref - cell_temperature))
            / ((cell_temperature - 254.33) * (temperature_ref - 254.33))
        )
        soc_effect = (
            2.62771800e-11 * dod**5.0
            - 1.48987233e-08 * dod**4.0
            + 2.03615618e-06 * dod**3.0
            - 1.06451730e-04 * dod**2.0
            + 2.13818712e-03 * dod
            + 3.90444549e-02
        )

        partials["internal_resistance", "state_of_charge"] = -(
            (
                5.0 * 2.62771800e-11 * dod**4.0
                - 4.0 * 1.48987233e-08 * dod**3.0
                + 3.0 * 2.03615618e-06 * dod**2.0
                - 2.0 * 1.06451730e-04 * dod
                + 2.13818712e-03
            )
            * temperature_effect
        )
        partials["internal_resistance", "cell_temperature"] = -(
            soc_effect * temperature_effect * 46.39 / (cell_temperature - 254.33) ** 2.0
        )
        partials[
            "internal_resistance",
            "settings:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":reference_temperature",
        ] = soc_effect * temperature_effect * 46.39 / (temperature_ref - 254.33) ** 2.0
