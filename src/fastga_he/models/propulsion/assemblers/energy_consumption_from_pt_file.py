# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator


class EnergyConsumptionFromPTFile(om.ExplicitComponent):
    """
    Assembles the contribution of all source to the aircraft level energy consumption.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])

        source_names = self.configurator.get_energy_consumption_list()

        number_of_points = self.options["number_of_points"]

        self.add_output(
            "fuel_consumed_t_econ",
            val=np.full(number_of_points, 0.0),
            desc="fuel consumed at each time step",
            units="kg",
        )
        self.add_output(
            "non_consumable_energy_t_econ",
            val=np.full(number_of_points, 0.0),
            desc="Non consumable energy at each time step",
            units="W*h",
        )

        for source_name in source_names:
            self.add_input(
                source_name + "_fuel_consumed_t",
                units="kg",
                val=np.full(number_of_points, np.nan),
            )
            self.declare_partials(
                of="fuel_consumed_t_econ",
                wrt=source_name + "_fuel_consumed_t",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

            self.add_input(
                source_name + "_non_consumable_energy_t",
                units="W*h",
                val=np.full(number_of_points, np.nan),
            )
            self.declare_partials(
                of="non_consumable_energy_t_econ",
                wrt=source_name + "_non_consumable_energy_t",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        source_names = self.configurator.get_energy_consumption_list()

        fuel_consumed = np.zeros(number_of_points)
        energy_consumed = np.zeros(number_of_points)

        for source_name in source_names:
            fuel_consumed += inputs[source_name + "_fuel_consumed_t"]
            energy_consumed += inputs[source_name + "_non_consumable_energy_t"]

        outputs["fuel_consumed_t_econ"] = fuel_consumed
        outputs["non_consumable_energy_t_econ"] = energy_consumed
