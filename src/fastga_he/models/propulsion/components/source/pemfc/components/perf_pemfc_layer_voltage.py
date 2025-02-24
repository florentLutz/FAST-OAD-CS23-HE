# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE
from .perf_ambient_pressure import PerformancesPEMFCStackAmbientPressure
from .perf_pemfc_operating_pressure import PerformancesPEMFCStackOperatingPressure
from .perf_pemfc_operating_temperature import PerformancesPEMFCStackOperatingTemperature
from .perf_pemfc_voltage_adjustment import PerformancesPEMFCStackVoltageAdjustment
from .perf_pemfc_polarization_curve import (
    PerformancesPEMFCStackPolarizationCurveSimple,
    PerformancesPEMFCStackPolarizationCurveAnalytical,
)

oad.RegisterSubmodel.active_models[SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE] = (
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple"
)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple",
)
class PerformancesPEMFCStackSingleVoltageSimple(om.Group):
    """
    This classes groups all the required calculation to obtain single layer voltage of PEMFC for
    simple model.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC operation pressure have to adjust based on compressor connection for "
            "oxygen inlet",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        max_current_density = self.options["max_current_density"]
        compressor_connection = self.options["compressor_connection"]

        self.add_subsystem(
            "ambient_pressure",
            PerformancesPEMFCStackAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_operating_pressure",
            PerformancesPEMFCStackOperatingPressure(
                number_of_points=number_of_points, compressor_connection=compressor_connection
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "polarization_curve",
            PerformancesPEMFCStackPolarizationCurveSimple(
                number_of_points=number_of_points,
                max_current_density=max_current_density,
            ),
            promotes=["*"],
        )


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.analytical",
)
class PerformancesPEMFCStackSingleVoltageAnalytical(om.Group):
    """
    This classes groups all the required calculation to obtain single layer voltage of PEMFC for
    analytical model.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC operation pressure have to adjust based on compressor connection for "
            "oxygen inlet",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        max_current_density = self.options["max_current_density"]
        compressor_connection = self.options["compressor_connection"]

        self.add_subsystem(
            "ambient_pressure",
            PerformancesPEMFCStackAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_operating_pressure",
            PerformancesPEMFCStackOperatingPressure(
                number_of_points=number_of_points, compressor_connection=compressor_connection
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_voltage_adjustment",
            PerformancesPEMFCStackVoltageAdjustment(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_ambient_temperature",
            PerformancesPEMFCStackOperatingTemperature(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "polarization_curve",
            PerformancesPEMFCStackPolarizationCurveAnalytical(
                pemfc_stack_id=pemfc_stack_id,
                number_of_points=number_of_points,
                max_current_density=max_current_density,
            ),
            promotes=["*"],
        )
