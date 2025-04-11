# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAirframeMaintenanceCost(om.ExplicitComponent):
    """
    Compute the annual cost of the airframe maintenance, obtained from :cite:`salgas:2025`.
    """

    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        self.add_input(
            "data:cost:operation:airframe_labor_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual airframe maintenance labor cost per aircraft",
        )

        self.add_input(
            "data:cost:operation:airframe_material_cost",
            val=np.nan,
            units="USD/yr",
            desc="Annual airframe maintenance material cost per aircraft",
        )

        self.add_output(
            "data:cost:operation:airframe_maintenance_cost",
            val=2.0e4,
            units="USD/yr",
            desc="Annual airframe maintenance cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:airframe_maintenance_cost"] = (
            inputs["data:cost:operation:airframe_labor_cost"]
            + inputs["data:cost:operation:airframe_material_cost"]
        )
