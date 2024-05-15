# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_current_rms_3_phases import PerformancesCurrentRMS3Phases
from ..components.perf_torque import PerformancesTorque
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum


class PerformancesTurboGenerator(om.Group):
    """
    Module that evaluates the performances of a turbo generator. Assumes fixed efficiency and
    power factor.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):

        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            name="rpm_mission",
            subsys=PerformancesRPMMission(
                turbo_generator_id=turbo_generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="voltage_target",
            subsys=PerformancesVoltageOutTargetMission(
                turbo_generator_id=turbo_generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="voltage_rms",
            subsys=PerformancesVoltageRMS(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="rms_current_3_phases",
            subsys=PerformancesCurrentRMS3Phases(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="apparent_power",
            subsys=PerformancesApparentPower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="active_power",
            subsys=PerformancesActivePower(
                turbo_generator_id=turbo_generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_in",
            subsys=PerformancesShaftPowerIn(
                turbo_generator_id=turbo_generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="torque",
            subsys=PerformancesTorque(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="voltage_peak",
            subsys=PerformancesVoltagePeak(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(
                turbo_generator_id=turbo_generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
