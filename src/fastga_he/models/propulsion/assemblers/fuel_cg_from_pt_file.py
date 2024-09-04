# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from fastga_he.powertrain_builder.powertrain import PT_DATA_PREFIX


class FuelCGFromPTFile(om.ExplicitComponent):
    """
    Assembles the contribution of all the fuel in the different tanks to the CG of the aircraft.
    Looks like a cleaner way to do things if there are more than two tanks in the wing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

        self._tank_names = None
        self._tank_types = None

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

        self._tank_names, self._tank_types = self.configurator.get_fuel_tank_list()

        number_of_points = self.options["number_of_points"]

        # Need an input in case there is no tank, this one seems like the most appropriate one
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output(
            "fuel_lever_arm_t_econ",
            val=np.linspace(600.0, 0.0, number_of_points),
            desc="lever arm of the fuel in the various tanks",
            units="kg*m",
        )
        self.add_output(
            "fuel_mass_t_econ",
            val=np.linspace(200.0, 0.0, number_of_points),
            desc="mass of the fuel in the various tanks",
            units="kg",
        )

        for tank_name, tank_type in zip(self._tank_names, self._tank_types):
            self.add_input(
                tank_name + "_fuel_remaining_t",
                units="kg",
                val=np.full(number_of_points, np.nan),
            )
            self.declare_partials(
                of="fuel_mass_t_econ",
                wrt=tank_name + "_fuel_remaining_t",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )
            self.declare_partials(
                of="fuel_lever_arm_t_econ",
                wrt=tank_name + "_fuel_remaining_t",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                method="exact",
            )

            self.add_input(
                PT_DATA_PREFIX + tank_type + ":" + tank_name + ":CG:x",
                units="m",
                val=np.nan,
            )
            self.declare_partials(
                of="fuel_lever_arm_t_econ",
                wrt=PT_DATA_PREFIX + tank_type + ":" + tank_name + ":CG:x",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        fuel_mass_t_econ = np.zeros(number_of_points)
        fuel_lever_arm_t_econ = np.zeros(number_of_points)

        for tank_name, tank_type in zip(self._tank_names, self._tank_types):
            fuel_mass_t_econ += inputs[tank_name + "_fuel_remaining_t"]
            fuel_lever_arm_t_econ += (
                inputs[tank_name + "_fuel_remaining_t"]
                * inputs[PT_DATA_PREFIX + tank_type + ":" + tank_name + ":CG:x"]
            )

        outputs["fuel_mass_t_econ"] = fuel_mass_t_econ
        outputs["fuel_lever_arm_t_econ"] = fuel_lever_arm_t_econ

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        for tank_name, tank_type in zip(self._tank_names, self._tank_types):
            partials["fuel_lever_arm_t_econ", tank_name + "_fuel_remaining_t"] = np.full(
                number_of_points, inputs[PT_DATA_PREFIX + tank_type + ":" + tank_name + ":CG:x"]
            )
            partials[
                "fuel_lever_arm_t_econ", PT_DATA_PREFIX + tank_type + ":" + tank_name + ":CG:x"
            ] = inputs[tank_name + "_fuel_remaining_t"]
