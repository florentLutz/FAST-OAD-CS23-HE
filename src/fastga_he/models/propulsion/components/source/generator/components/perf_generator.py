# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_current_rms_3_phases import PerformancesCurrentRMS3Phases
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum


class PerformancesGenerator(om.Group):
    """
    Module that evaluates the performances of a generator, based on the PMSM model itself based on
    regressions on the EMRAX family. The reason behind that choice is that some papers have
    described the use of EMRAX motor as a generator in a configuration coupled to an ICE, see
    :cite:`geiss:2018` and :cite:`kalwara:2021`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):

        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            name="rpm_mission",
            subsys=PerformancesRPMMission(
                generator_id=generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="rms_current_3_phases",
            subsys=PerformancesCurrentRMS3Phases(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="torque",
            subsys=PerformancesTorque(generator_id=generator_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="losses",
            subsys=PerformancesLosses(generator_id=generator_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_in",
            subsys=PerformancesShaftPowerIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="efficiency",
            subsys=PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="active_power",
            subsys=PerformancesActivePower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="apparent_power",
            subsys=PerformancesApparentPower(
                generator_id=generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="voltage_rms",
            subsys=PerformancesVoltageRMS(number_of_points=number_of_points),
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
                generator_id=generator_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
