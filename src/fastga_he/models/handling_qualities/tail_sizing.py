# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
# Temporary fix for certain cases

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem(
    "fastga_he.handling_qualities.fixed_tail_sizing", domain=ModelDomain.HANDLING_QUALITIES
)
class ComputeTailAreas(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:volumetric_coefficient", val=np.nan)
        self.add_input("data:geometry:vertical_tail:volumetric_coefficient", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_output("data:geometry:horizontal_tail:area", val=4.0, units="m**2")
        self.add_output("data:geometry:vertical_tail:area", val=4.0, units="m**2")

    def setup_partials(self):
        self.declare_partials(
            of="data:geometry:horizontal_tail:area",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:MAC:length",
                "data:geometry:horizontal_tail:volumetric_coefficient",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ],
        )
        self.declare_partials(
            of="data:geometry:vertical_tail:area",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:span",
                "data:geometry:vertical_tail:volumetric_coefficient",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_wing = inputs["data:geometry:wing:area"]
        b_wing = inputs["data:geometry:wing:span"]
        mac_wing = inputs["data:geometry:wing:MAC:length"]
        l_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        vc_ht = inputs["data:geometry:horizontal_tail:volumetric_coefficient"]
        vc_vt = inputs["data:geometry:vertical_tail:volumetric_coefficient"]

        outputs["data:geometry:horizontal_tail:area"] = vc_ht * s_wing * mac_wing / l_ht
        outputs["data:geometry:vertical_tail:area"] = vc_vt * s_wing * b_wing / l_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_wing = inputs["data:geometry:wing:area"]
        b_wing = inputs["data:geometry:wing:span"]
        mac_wing = inputs["data:geometry:wing:MAC:length"]
        l_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        vc_ht = inputs["data:geometry:horizontal_tail:volumetric_coefficient"]
        vc_vt = inputs["data:geometry:vertical_tail:volumetric_coefficient"]

        partials["data:geometry:horizontal_tail:area", "data:geometry:wing:area"] = (
            vc_ht * mac_wing / l_ht
        )

        partials["data:geometry:horizontal_tail:area", "data:geometry:wing:MAC:length"] = (
            vc_ht * s_wing / l_ht
        )

        partials[
            "data:geometry:horizontal_tail:area",
            "data:geometry:horizontal_tail:volumetric_coefficient",
        ] = s_wing * mac_wing / l_ht

        partials[
            "data:geometry:horizontal_tail:area",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = -vc_ht * s_wing * mac_wing / l_ht**2.0

        partials["data:geometry:vertical_tail:area", "data:geometry:wing:area"] = (
            vc_vt * b_wing / l_vt
        )

        partials["data:geometry:vertical_tail:area", "data:geometry:wing:span"] = (
            vc_vt * s_wing / l_vt
        )

        partials[
            "data:geometry:vertical_tail:area", "data:geometry:vertical_tail:volumetric_coefficient"
        ] = s_wing * b_wing / l_vt

        partials[
            "data:geometry:vertical_tail:area",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = -vc_vt * s_wing * b_wing / l_vt**2.0
