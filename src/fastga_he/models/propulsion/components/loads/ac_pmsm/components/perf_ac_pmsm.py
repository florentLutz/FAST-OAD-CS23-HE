# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .perf_torque import PerformancesTorque
from .perf_iron_losses import PerformancesIronLosses
from .perf_joule_losses import PerformancesJouleLosses
from .perf_mechanical_losses import PerformancesMechanicalLosses
from .perf_power_losses import PerformancesPowerLosses
from .perf_efficiency import PerformancesEfficiency
from .perf_frequency import PerformancesFrequency
from .perf_active_power import PerformancesActivePower
from .perf_apparent_power import PerformancesApparentPower
from .perf_current_rms import PerformancesCurrentRMS
from .perf_current_rms_phase import PerformancesCurrentRMS1Phase
from .perf_voltage_rms import PerformancesVoltageRMS
from .perf_windage_reynolds import PerformancesWindageReynolds
from .perf_windage_friction_coeff import PerformancesWindageFrictionCoefficient
from .perf_voltage_peak import PerformancesVoltagePeak
from .perf_maximum import PerformancesMaximum


class PerformancesACPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "torque",
            PerformancesTorque(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "windage_reynold",
            PerformancesWindageReynolds(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "windage_friction_coeff",
            PerformancesWindageFrictionCoefficient(
                pmsm_id=pmsm_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "frequency",
            PerformancesFrequency(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "current_rms",
            PerformancesCurrentRMS(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "current_rms_one_phase",
            PerformancesCurrentRMS1Phase(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "iron_losses",
            PerformancesIronLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "joule_losses",
            PerformancesJouleLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "mechanical_losses",
            PerformancesMechanicalLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "power_losses",
            PerformancesPowerLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "active_power",
            PerformancesActivePower(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "apparent_power",
            PerformancesApparentPower(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "voltage_rms",
            PerformancesVoltageRMS(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "voltage_peak",
            PerformancesVoltagePeak(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, pmsm_id=pmsm_id),
            promotes=["*"],
        )
