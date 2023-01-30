# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_energy_coefficient_scaling import (
    SizingDCDCConverterEnergyCoefficientScaling,
)
from .sizing_energy_coefficients import SizingDCDCConverterEnergyCoefficients
from .sizing_resistance_scaling import SizingDCDCConverterResistanceScaling
from .sizing_reference_resistance import SizingDCDCConverterResistances
from .sizing_inductor_inductance import SizingDCDCConverterInductorInductance
from .sizing_capacitor_capacity import SizingDCDCConverterCapacitorCapacity
from .sizing_weight import SizingDCDCConverterWeight
from .sizing_dc_dc_converter_cg import SizingDCDCConverterCG
from .sizing_dc_dc_converter_drag import SizingDCDCConverterDrag

from .cstr_dc_dc_converter import ConstraintsDCDCConverter

from ..constants import POSSIBLE_POSITION


class SizingDCDCConverter(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC/DC converter.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the inverter, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        position = self.options["position"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_dc_dc_converter",
            subsys=ConstraintsDCDCConverter(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "energy_coefficients_scaling",
            SizingDCDCConverterEnergyCoefficientScaling(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "energy_coefficients",
            SizingDCDCConverterEnergyCoefficients(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "resistance_scaling",
            SizingDCDCConverterResistanceScaling(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "resistance",
            SizingDCDCConverterResistances(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "capacitor_capacity",
            SizingDCDCConverterCapacitorCapacity(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "inductor_inductance",
            SizingDCDCConverterInductorInductance(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "converter_weight",
            SizingDCDCConverterWeight(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="converter_CG",
            subsys=SizingDCDCConverterCG(dc_dc_converter_id=dc_dc_converter_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "converter_drag_ls" if low_speed_aero else "converter_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingDCDCConverterDrag(
                    dc_dc_converter_id=dc_dc_converter_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
