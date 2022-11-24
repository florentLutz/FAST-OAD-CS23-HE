# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER

from ..components.sizing_energy_coefficient_scaling import (
    SizingDCDCConverterEnergyCoefficientScaling,
)
from ..components.sizing_energy_coefficients import SizingDCDCConverterEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingDCDCConverterResistanceScaling
from ..components.sizing_reference_resistance import SizingDCDCConverterResistances
from ..components.sizing_inductor_inductance import SizingDCDCConverterInductorInductance
from ..components.sizing_capacitor_capacity import SizingDCDCConverterCapacitorCapacity
from ..components.sizing_weight import SizingDCDCConverterWeight


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

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_dc_dc_converter_id = {"dc_dc_converter_id": dc_dc_converter_id}

        self.add_subsystem(
            name="constraints_dc_dc_converter",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER, options=option_dc_dc_converter_id
            ),
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
