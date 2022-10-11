# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_torque import PerformancesTorque
from .perf_losses import PerformancesLosses
from .perf_efficiency import PerformancesEfficiency
from .perf_active_power import PerformancesActivePower
from .perf_apparent_power import PerformancesApparentPower
from .perf_current_rms import PerformancesCurrentRMS
from .perf_current_rms_phase import PerformancesCurrentRMS1Phase
from .perf_voltage_rms import PerformancesVoltageRMS
from .perf_voltage_peak import PerformancesVoltagePeak


class PerformancePMSM(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "torque",
            PerformancesTorque(number_of_points=number_of_points),
            promotes=["shaft_power", "rpm"],
        )

        self.add_subsystem(
            "losses",
            PerformancesLosses(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["data:*", "rpm"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["shaft_power"],
        )
        self.add_subsystem(
            "active_power",
            PerformancesActivePower(number_of_points=number_of_points),
            promotes=["shaft_power"],
        )
        self.add_subsystem(
            "apparent_power",
            PerformancesApparentPower(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["settings:*"],
        )

        self.add_subsystem(
            "rms_current",
            PerformancesCurrentRMS(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["data:*", "rms_current"],
        )
        self.add_subsystem(
            "rms_current_one_phase",
            PerformancesCurrentRMS1Phase(number_of_points=number_of_points),
            promotes=["rms_current", "rms_current_one_phase"],
        )
        self.add_subsystem(
            "rms_voltage",
            PerformancesVoltageRMS(number_of_points=number_of_points),
            promotes=["rms_current", "rms_voltage"],
        )
        self.add_subsystem(
            "peak_voltage",
            PerformancesVoltagePeak(number_of_points=number_of_points),
            promotes=["peak_voltage", "rms_voltage"],
        )

        self.connect(
            "torque.torque",
            ["losses.torque", "rms_current.torque"],
        )

        self.connect(
            "losses.power_losses",
            "efficiency.power_losses",
        )

        self.connect(
            "efficiency.efficiency",
            "active_power.efficiency",
        )

        self.connect(
            "active_power.active_power",
            "apparent_power.active_power",
        )

        self.connect(
            "apparent_power.apparent_power",
            "rms_voltage.apparent_power",
        )
