# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .perf_iron_losses import PerformancesIronLosses
from .perf_joule_losses import PerformancesJouleLosses
from .perf_windage_reynolds import PerformancesWindageReynolds
from .perf_windage_friction_coeff import PerformancesWindageFrictionCoefficient
from .perf_airgap_windage_loss import PerformancesAirgapWindageLoss
from .perf_rotor_windage_loss import PerformancesRotorWindageLoss
from .perf_bearing_friction_loss import PerformancesBearingLoss
from .perf_mechanical_losses import PerformancesMechanicalLosses
from .perf_power_losses import PerformancesPowerLosses
from .perf_efficiency import PerformancesEfficiency
from .perf_apparent_power import PerformancesApparentPower
from .perf_current_rms import PerformancesCurrentRMS
from .perf_maximum import PerformancesMaximum
from .perf_electrical_frequency import PerformancesElectricalFrequency
from ...pmsm.components.perf_torque import PerformancesTorque
from ...pmsm.components.perf_active_power import PerformancesActivePower
from ...pmsm.components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ...pmsm.components.perf_voltage_rms import PerformancesVoltageRMS
from ...pmsm.components.perf_voltage_peak import PerformancesVoltagePeak


class PerformancesSMPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "torque",
            PerformancesTorque(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "windage_reynold",
            PerformancesWindageReynolds(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "windage_friction_coeff",
            PerformancesWindageFrictionCoefficient(
                motor_id=motor_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "electrical_frequency",
            PerformancesElectricalFrequency(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "current_rms",
            PerformancesCurrentRMS(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "current_rms_one_phase",
            PerformancesCurrentRMS1Phase(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "iron_losses",
            PerformancesIronLosses(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "joule_losses",
            PerformancesJouleLosses(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "airgap_windage_loss",
            PerformancesAirgapWindageLoss(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "rotor_windage_loss",
            PerformancesRotorWindageLoss(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "bearing_friction_loss",
            PerformancesBearingLoss(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "mechanical_losses",
            PerformancesMechanicalLosses(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "power_losses",
            PerformancesPowerLosses(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(motor_id=motor_id, number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "active_power",
            PerformancesActivePower(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            "apparent_power",
            PerformancesApparentPower(motor_id=motor_id, number_of_points=number_of_points),
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
            PerformancesMaximum(number_of_points=number_of_points, motor_id=motor_id),
            promotes=["*"],
        )
