# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_ambient_pressure import PerformancesPEMFCStackBOPAmbientPressure
from .perf_pemfc_operating_pressure import PerformancesPEMFCStackBOPOperatingPressure
from .perf_pemfc_voltage_adjustment import PerformancesPEMFCStackBOPVoltageAdjustment
from .perf_pemfc_polarization_curve import (
    PerformancesPEMFCStackBOPPolarizationCurveEmpirical,
    PerformancesPEMFCStackBOPPolarizationCurveAnalytical,
)


class PerformancesPEMFCStackBOPSingleLayerVoltageEmpirical(om.Group):
    """
    This class groups all the required calculation to obtain single layer voltage of the PEMFC
    stack for the empirical model.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air inlet",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]
        compressor_connection = self.options["compressor_connection"]

        self.add_subsystem(
            "ambient_pressure",
            PerformancesPEMFCStackBOPAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_operating_pressure",
            PerformancesPEMFCStackBOPOperatingPressure(
                number_of_points=number_of_points,
                compressor_connection=compressor_connection,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "polarization_curve",
            PerformancesPEMFCStackBOPPolarizationCurveEmpirical(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )


class PerformancesPEMFCStackBOPSingleLayerVoltageAnalytical(om.Group):
    """
    This class groups all the required calculation to obtain single layer voltage of the PEMFC
    stack for the analytical model.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air inlet",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]
        compressor_connection = self.options["compressor_connection"]

        self.add_subsystem(
            "ambient_pressure",
            PerformancesPEMFCStackBOPAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_operating_pressure",
            PerformancesPEMFCStackBOPOperatingPressure(
                number_of_points=number_of_points,
                compressor_connection=compressor_connection,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_voltage_adjustment",
            PerformancesPEMFCStackBOPVoltageAdjustment(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "polarization_curve",
            PerformancesPEMFCStackBOPPolarizationCurveAnalytical(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
