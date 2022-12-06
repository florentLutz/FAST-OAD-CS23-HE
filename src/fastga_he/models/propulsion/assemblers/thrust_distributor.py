# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_THRUST_DISTRIBUTOR


@oad.RegisterSubmodel(
    SUBMODEL_THRUST_DISTRIBUTOR, "fastga_he.submodel.propulsion.thrust_distributor.legacy"
)
class ThrustDistributor(om.ExplicitComponent):
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

        propulsor_names = self.configurator.get_thrust_element_list()

        number_of_points = self.options["number_of_points"]
        number_of_propulsor = len(propulsor_names)

        self.add_input("thrust", units="N", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:thrust_distribution",
            val=np.full(number_of_propulsor, 1.0),
        )

        for propulsor_name in propulsor_names:
            self.add_output(
                propulsor_name + "_thrust", units="N", val=np.full(number_of_points, 250.0)
            )

            self.declare_partials(
                of=propulsor_name + "_thrust",
                wrt=["thrust", "data:propulsion:he_power_train:thrust_distribution"],
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsor_names = self.configurator.get_thrust_element_list()

        # First we rescale the thrust distribution so that at all point it is between 0 and 1
        thrust_distribution = inputs["data:propulsion:he_power_train:thrust_distribution"] / np.sum(
            inputs["data:propulsion:he_power_train:thrust_distribution"]
        )
        aircraft_thrust = inputs["thrust"]

        for propulsor_name in propulsor_names:
            outputs[propulsor_name + "_thrust"] = (
                aircraft_thrust * thrust_distribution[propulsor_names.index(propulsor_name)]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propulsor_names = self.configurator.get_thrust_element_list()

        number_of_points = self.options["number_of_points"]
        number_of_propulsor = len(propulsor_names)

        # First we rescale the thrust distribution so that at all point it is between 0 and 1

        scale_factor = np.sum(inputs["data:propulsion:he_power_train:thrust_distribution"])
        thrust_distribution = (
            inputs["data:propulsion:he_power_train:thrust_distribution"] / scale_factor
        )
        aircraft_thrust = inputs["thrust"]

        for propulsor_name in propulsor_names:
            partials[propulsor_name + "_thrust", "thrust"] = (
                np.eye(number_of_points)
                * thrust_distribution[propulsor_names.index(propulsor_name)]
            )

            base_partials = (
                (-np.tile(aircraft_thrust, (number_of_propulsor, 1)))
                * thrust_distribution[propulsor_names.index(propulsor_name)]
                / scale_factor
            )
            base_partials[propulsor_names.index(propulsor_name), :] = (
                aircraft_thrust
                * (1.0 - thrust_distribution[propulsor_names.index(propulsor_name)])
                / scale_factor
            )
            partials[
                propulsor_name + "_thrust", "data:propulsion:he_power_train:thrust_distribution"
            ] = np.transpose(base_partials)
