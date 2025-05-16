# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDailyParkingCost(om.ExplicitComponent):
    """
    Computation of the aircraft daily parking cost.
    Prices provide from
    https://www.toulouse.aeroport.fr/sites/default/files/2023-03/2023_03_23_Redevances_A_EN_v1.pdf.
    """

    def setup(self):
        self.add_input(
            "data:weight:aircraft:MTOW",
            units="t",
            val=np.nan,
        )
        self.add_input(
            "data:cost:operation:parking_fee_per_ton",
            val=11.8,
            units="USD",
            desc="Daily parking fee per tons, apply for MTOW > 6.0 tons",
        )

        self.add_output(
            "data:cost:operation:daily_parking_cost",
            val=15.0,
            units="USD/d",
            desc="Daily parking cost of the aircraft",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        cost_per_ton = inputs["data:cost:operation:parking_fee_per_ton"]

        if mtow > 6.0:
            daily_parking_cost = mtow * cost_per_ton
        elif 2.5 < mtow <= 6.0:
            daily_parking_cost = 7.2
        elif 1.5 < mtow <= 2.5:
            daily_parking_cost = 4.186
        else:
            daily_parking_cost = 2.132

        outputs["data:cost:operation:daily_parking_cost"] = daily_parking_cost

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if inputs["data:weight:aircraft:MTOW"] > 6.0:
            partials["data:cost:operation:daily_parking_cost", "data:weight:aircraft:MTOW"] = (
                inputs["data:cost:operation:parking_fee_per_ton"]
            )
            partials[
                "data:cost:operation:daily_parking_cost", "data:cost:operation:parking_fee_per_ton"
            ] = inputs["data:weight:aircraft:MTOW"]
