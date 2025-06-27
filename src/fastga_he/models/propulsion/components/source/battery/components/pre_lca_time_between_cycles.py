#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryTimeBetweenCycles(om.ExplicitComponent):
    """
    Since the battery life was chosen to be expressed in terms of cycles but for calendar aging we
    need time, we need to express how much time is spent between cycle and for 1 cycle. This will
    be based on the average usage of the aircraft. It will be expressed in days. Also, we'll assume
    that one flight is one cycle
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":time_between_cycle",
            units="d",  # This is the unit for days
            val=10.0,
            desc="Amount of time between two cycles of the battery, used for calendar aging computation",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":time_between_cycle"
        ] = 365.0 / inputs["data:TLAR:flight_per_year"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":time_between_cycle",
            "data:TLAR:flight_per_year",
        ] = -365.0 / inputs["data:TLAR:flight_per_year"] ** 2.0
