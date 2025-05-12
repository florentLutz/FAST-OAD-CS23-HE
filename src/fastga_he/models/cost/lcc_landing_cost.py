# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLandingCost(om.ExplicitComponent):
    """
    Computation of the cost related to aircraft landing. Prices obtained from
    https://www.toulouse.aeroport.fr/sites/default/files/2023-03/2023_03_23_Redevances_A_EN_v1.pdf.
    """

    def setup(self):
        self.add_input(
            "data:weight:aircraft:MTOW",
            units="t",
            val=np.nan,
        )

        self.add_input(
            "data:cost:operation:airport_charge_VAT",
            val=0.2,
            desc="VAT cost of the airport fees",
        )

        self.add_output(
            "data:cost:operation:landing_cost",
            val=60.0,
            units="USD",
            desc="Landing cost per operation of the aircraft",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        vat = inputs["data:cost:operation:airport_charge_VAT"]

        if mtow >= 7.0:
            landing_cost_ex = 54.88 + 0.56 * (mtow - 6.0)
        elif 2.5 < mtow < 7.0:
            landing_cost_ex = 60.27
        elif 1.5 < mtow <= 2.5:
            landing_cost_ex = 44.83
        else:
            landing_cost_ex = 34.66

        outputs["data:cost:operation:landing_cost"] = landing_cost_ex * (1.0 + vat)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        vat = inputs["data:cost:operation:airport_charge_VAT"]

        if mtow >= 7.0:
            partials["data:cost:operation:landing_cost", "data:weight:aircraft:MTOW"] = 0.56 * (
                1.0 + vat
            )
            landing_cost_ex = 54.88 + 0.56 * (mtow - 6.0)
        elif 2.5 < mtow < 7.0:
            landing_cost_ex = 60.27
        elif 1.5 < mtow <= 2.5:
            landing_cost_ex = 44.83
        else:
            landing_cost_ex = 34.66

        partials["data:cost:operation:landing_cost", "data:cost:operation:airport_charge_VAT"] = (
            landing_cost_ex
        )
