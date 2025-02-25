# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .perf_ambient_pressure import PerformancesPEMFCStackAmbientPressure
from .perf_pemfc_operating_pressure import PerformancesPEMFCStackOperatingPressure
from .perf_pemfc_operating_temperature import PerformancesPEMFCStackOperatingTemperature
from .perf_pemfc_voltage_adjustment import PerformancesPEMFCStackVoltageAdjustment
from .perf_pemfc_polarization_curve import (
    PerformancesPEMFCStackPolarizationCurveEmpirical,
    PerformancesPEMFCStackPolarizationCurveAnalytical,
)


class PerformancesPEMFCStackSingleVoltageEmpirical(om.Group):
    """
    This classes groups all the required calculation to obtain single layer voltage of PEMFC for
    empirical model.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
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
            PerformancesPEMFCStackPolarizationCurveEmpirical(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
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
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC operation pressure have to adjust based on compressor connection for "
            "oxygen inlet",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]
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
            ),
            promotes=["*"],
        )
