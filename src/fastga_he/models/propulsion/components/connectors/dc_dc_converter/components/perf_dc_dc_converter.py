# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_converter_relations import PerformancesConverterRelations
from .perf_generator_side import PerformancesConverterGeneratorSide
from .perf_load_side import PerformancesConverterLoadSide


class PerformancesDCDCConverter(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("efficiency", default=0.98, desc="Converter efficiency")
        self.options.declare("voltage_target", desc="Target voltage")

    def setup(self):
        efficiency = self.options["efficiency"]
        voltage_target = self.options["voltage_target"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "load_side",
            PerformancesConverterLoadSide(number_of_points=number_of_points),
            promotes=["voltage_in", "current_in"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesConverterGeneratorSide(number_of_points=number_of_points),
            promotes=["voltage_out", "current_out"],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesConverterRelations(
                voltage_target=voltage_target,
                efficiency=efficiency,
                number_of_points=number_of_points,
            ),
            promotes=["voltage_out"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("current_out", "converter_relation.current_out")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")
