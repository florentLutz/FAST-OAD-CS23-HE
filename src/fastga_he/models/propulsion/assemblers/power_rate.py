# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_RATE


@oad.RegisterSubmodel(SUBMODEL_POWER_RATE, "fastga_he.submodel.propulsion.power_rate.legacy")
class PowerRate(om.ExplicitComponent):
    """
    To work with the current version of the mission vector, a mock-up interface is required to
    provide the computation of the thrust rate. Rather than giving a fake number a power rate
    will be computed based on the max shaft power during the mission and the shaft power at each
    point. This component will also be used to plug in the input provided by the mission that are
    not used by the power train performances computation.
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

        (
            propulsive_load_names,
            propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        number_of_points = self.options["number_of_points"]

        # Now for the "mock-up" part
        self.add_input("altitude_econ", val=np.nan, shape=number_of_points, units="ft")
        self.add_input("density_econ", val=np.nan, shape=number_of_points, units="kg/m**3")
        self.add_input("engine_setting_econ", val=np.nan, shape=number_of_points)
        self.add_input(
            "exterior_temperature_econ", val=273.15, shape=number_of_points, units="degK"
        )

        self.add_output("thrust_rate_t_econ", val=np.ones(number_of_points))

        for propulsive_load_name, propulsive_load_type in zip(
            propulsive_load_names,
            propulsive_load_types,
        ):
            self.add_input(
                "data:propulsion:he_power_train:"
                + propulsive_load_type
                + ":"
                + propulsive_load_name
                + ":shaft_power_rating",
                units="W",
                val=np.nan,
            )
            self.add_input(
                propulsive_load_name + "_shaft_power_out",
                units="W",
                shape=number_of_points,
                val=np.nan,
            )
            self.declare_partials(
                of="thrust_rate_t_econ",
                wrt="data:propulsion:he_power_train:"
                + propulsive_load_type
                + ":"
                + propulsive_load_name
                + ":shaft_power_rating",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )
            self.declare_partials(
                of="thrust_rate_t_econ",
                wrt=propulsive_load_name + "_shaft_power_out",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        (
            propulsive_load_names,
            propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        number_of_points = self.options["number_of_points"]

        max_power = 0
        mission_power = np.zeros(number_of_points)

        for propulsive_load_name, propulsive_load_type in zip(
            propulsive_load_names,
            propulsive_load_types,
        ):
            max_power += inputs[
                "data:propulsion:he_power_train:"
                + propulsive_load_type
                + ":"
                + propulsive_load_name
                + ":shaft_power_rating"
            ]
            mission_power += inputs[propulsive_load_name + "_shaft_power_out"]

        outputs["thrust_rate_t_econ"] = mission_power / max_power

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        (
            propulsive_load_names,
            propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        number_of_points = self.options["number_of_points"]

        max_power = 0
        mission_power = np.zeros(number_of_points)

        for propulsive_load_name, propulsive_load_type in zip(
            propulsive_load_names,
            propulsive_load_types,
        ):
            max_power += inputs[
                "data:propulsion:he_power_train:"
                + propulsive_load_type
                + ":"
                + propulsive_load_name
                + ":shaft_power_rating"
            ]
            mission_power += inputs[propulsive_load_name + "_shaft_power_out"]

        for propulsive_load_name, propulsive_load_type in zip(
            propulsive_load_names,
            propulsive_load_types,
        ):
            partials["thrust_rate_t_econ", propulsive_load_name + "_shaft_power_out"] = (
                np.ones(number_of_points) / max_power
            )
            partials[
                "thrust_rate_t_econ",
                "data:propulsion:he_power_train:"
                + propulsive_load_type
                + ":"
                + propulsive_load_name
                + ":shaft_power_rating",
            ] = -mission_power / max_power**2.0
