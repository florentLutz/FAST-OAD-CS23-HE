"""Sum of form drag from aircraft components."""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastoad_cs25.models.aerodynamics.constants import SERVICE_CD0_SUM


@oad.RegisterSubmodel(SERVICE_CD0_SUM, "fastoad.submodel.aerodynamics.CD0.sum.rta")
class Cd0Total(om.Group):
    """Computes the sum of form drags from aircraft components."""

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            "parasitic_factor_" + ls_tag,
            _TotalCd0ParasiticFactor(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "cd0_aircraft_" + ls_tag,
            _AircraftCd0(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect(
            "parasitic_factor_" + ls_tag + ".k_parasitic", "cd0_aircraft_" + ls_tag + ".k_parasitic"
        )
        self.connect(
            "parasitic_factor_" + ls_tag + ".CD0_fuselage",
            "cd0_aircraft_" + ls_tag + ".CD0_fuselage",
        )


class _TotalCd0ParasiticFactor(om.ExplicitComponent):
    """
    Computes the parasitic factor of the total aircraft form drag and set fuselage CD0 to float.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:geometry:aircraft:wetted_area", val=np.nan, units="m**2")
        self.add_input(
            "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
            shape_by_conn=True,
            val=np.nan,
        )

        self.add_output("k_parasitic", val=0.1)
        self.add_output("CD0_fuselage", val=0.0005)

    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials("k_parasitic", "data:geometry:aircraft:wetted_area", method="exact")
        self.declare_partials(
            "CD0_fuselage", "data:aerodynamics:fuselage:" + ls_tag + ":CD0", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        wet_area_total = inputs["data:geometry:aircraft:wetted_area"]

        k_parasitic = (
            -2.39e-12 * wet_area_total**3.0
            + 2.58e-8 * wet_area_total**2.0
            - 0.89e-4 * wet_area_total
            + 0.163
        )

        outputs["k_parasitic"] = k_parasitic
        outputs["CD0_fuselage"] = np.median(inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        wet_area_total = inputs["data:geometry:aircraft:wetted_area"]
        cd0_fuselage = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]

        partials["k_parasitic", "data:geometry:aircraft:wetted_area"] = (
            -7.17e-12 * wet_area_total**2.0 + 5.16e-8 * wet_area_total - 0.89e-4
        )

        partials["CD0_fuselage", "data:aerodynamics:fuselage:" + ls_tag + ":CD0"] = np.where(
            cd0_fuselage == np.median(cd0_fuselage), 1.0, 0.0
        )


class _AircraftCd0(om.ExplicitComponent):
    """
    Computes the clean and parasitic aircraft form drag coefficient.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("k_parasitic")
        self.add_input("CD0_fuselage")
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:nacelles:" + ls_tag + ":CD0", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:" + ls_tag + ":CD0", val=0.022)
        self.add_output("data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean", val=0.02)
        self.add_output("data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic", val=0.002)

    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials(
            "*",
            [
                "data:aerodynamics:wing:" + ls_tag + ":CD0",
                "CD0_fuselage",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
            [
                "data:aerodynamics:wing:" + ls_tag + ":CD0",
                "CD0_fuselage",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
            ],
            val=1.0,
        )
        self.declare_partials(
            [
                "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
                "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            ],
            "k_parasitic",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        k_parasitic = inputs["k_parasitic"]
        cd0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        cd0_fus = inputs["CD0_fuselage"]
        cd0_ht = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0"]
        cd0_vt = inputs["data:aerodynamics:vertical_tail:" + ls_tag + ":CD0"]
        cd0_nac = inputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"]
        cd0_clean = cd0_wing + cd0_ht + cd0_vt + cd0_nac + cd0_fus

        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"] = cd0_clean
        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"] = k_parasitic * cd0_clean
        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0"] = (1.0 + k_parasitic) * cd0_clean

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        k_parasitic = inputs["k_parasitic"]
        cd0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        cd0_fus = inputs["CD0_fuselage"]
        cd0_ht = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0"]
        cd0_vt = inputs["data:aerodynamics:vertical_tail:" + ls_tag + ":CD0"]
        cd0_nac = inputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"]

        partials["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic", "k_parasitic"] = (
            cd0_wing + cd0_ht + cd0_vt + cd0_nac + cd0_fus
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:wing:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic", "CD0_fuselage"] = (
            k_parasitic
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials["data:aerodynamics:aircraft:" + ls_tag + ":CD0", "k_parasitic"] = (
            cd0_wing + cd0_ht + cd0_vt + cd0_nac + cd0_fus
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:wing:" + ls_tag + ":CD0",
        ] = 1.0 + k_parasitic

        partials["data:aerodynamics:aircraft:" + ls_tag + ":CD0", "CD0_fuselage"] = (
            1.0 + k_parasitic
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
        ] = 1.0 + k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
        ] = 1.0 + k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
        ] = 1.0 + k_parasitic
