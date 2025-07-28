# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_torque import PerformancesTorque
from .perf_iron_losses import PerformancesIronLosses
from .perf_Joule_losses_2 import PerformancesJouleLosses2
from .perf_mechanical_losses import PerformancesMechanicalLosses
from .perf_power_losses import PerformancesPowerLosses
from .perf_efficiency import PerformancesEfficiency
from .perf_active_power import PerformancesActivePower
from .perf_apparent_power import PerformancesApparentPower
from .perf_current_rms_new import PerformancesCurrentRMS
from .perf_current_rms_phase import PerformancesCurrentRMS1Phase
from .perf_voltage_rms import PerformancesVoltageRMS
from .perf_voltage_peak import PerformancesVoltagePeak
from .perf_maximum import PerformancesMaximum


class PerformancesACPMSMNEW(om.Group):
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
            "Ironlosses",
            PerformancesIronLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "Joulelosses",
            PerformancesJouleLosses2(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "Mechanicallosses",
            PerformancesMechanicalLosses(pmsm_id=pmsm_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "PowerLosses",
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
