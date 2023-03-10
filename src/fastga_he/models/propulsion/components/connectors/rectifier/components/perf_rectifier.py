# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_voltage_out_target import PerformancesVoltageOutTargetMission
from .perf_efficiency import PerformancesEfficiencyMission
from .perf_voltage_peak_in import PerformancesVoltagePeakIn
from .perf_modulation_index import PerformancesModulationIndex
from .perf_load_side import PerformancesRectifierLoadSide
from .perf_generator_side import PerformancesRectifierGeneratorSide
from .perf_rectifier_relations import PerformancesRectifierRelations
from .perf_maximum import PerformancesMaximum


class PerformancesRectifier(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "voltage_out_target",
            PerformancesVoltageOutTargetMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiencyMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "peak_voltage_in",
            PerformancesVoltagePeakIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "modulation_idx",
            PerformancesModulationIndex(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "load_side",
            PerformancesRectifierLoadSide(number_of_points=number_of_points),
            promotes=["ac_voltage_rms_in", "ac_current_rms_in_one_phase"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesRectifierGeneratorSide(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "dc_current_out"],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesRectifierRelations(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "voltage_out_target", "dc_current_out", "efficiency"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")
