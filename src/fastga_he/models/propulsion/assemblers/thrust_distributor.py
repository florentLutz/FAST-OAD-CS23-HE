# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_THRUST_DISTRIBUTOR

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_THRUST_DISTRIBUTOR, "fastga_he.submodel.propulsion.thrust_distributor.legacy"
)
class ThrustDistributor(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

        self.propulsor_names = None
        self.propulsor_connected = None

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

        self.propulsor_names = self.configurator.get_thrust_element_list()
        self.propulsor_connected = self.configurator.are_propulsor_connected_to_source()

        number_of_points = self.options["number_of_points"]
        number_of_propulsor = len(self.propulsor_names)

        self.add_input("thrust", units="N", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:thrust_distribution",
            val=np.full(number_of_propulsor, 1.0),
        )

        for propulsor_name in self.propulsor_names:
            self.add_output(
                propulsor_name + "_thrust", units="N", val=np.full(number_of_points, 250.0)
            )

            # First we check if the propulsor are actually connected to a source in which case
            # there is a link between the inputs and outputs. Otherwise, since we will overwrite
            # the contribution to zero to force it to not be used, there will be no dependency

            if self.propulsor_connected[propulsor_name]:
                self.declare_partials(
                    of=propulsor_name + "_thrust",
                    wrt="thrust",
                    method="exact",
                    rows=np.arange(number_of_points),
                    cols=np.arange(number_of_points),
                )
                self.declare_partials(
                    of=propulsor_name + "_thrust",
                    wrt="data:propulsion:he_power_train:thrust_distribution",
                    method="exact",
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        untouched_thrust_distributor = inputs["data:propulsion:he_power_train:thrust_distribution"]

        # First we force to zero all the propeller that are not connected if we are not using
        # temp copies as is the case for the advanced aerodynamic wing computation
        for propulsor_name in self.propulsor_names:
            if not self.propulsor_connected[propulsor_name] and not self.options[
                "power_train_file_path"
            ].endswith("temp_copy.yml"):
                untouched_thrust_distributor[self.propulsor_names.index(propulsor_name)] = 0
                _LOGGER.warning(
                    propulsor_name
                    + "is not connected to any source. Its contribution to the total aircraft "
                    "thrust was set to zero"
                )

        # Then we rescale the thrust distribution so that at all point it is between 0 and 1
        thrust_distribution = untouched_thrust_distributor / np.sum(untouched_thrust_distributor)
        aircraft_thrust = inputs["thrust"]

        for propulsor_name in self.propulsor_names:
            outputs[propulsor_name + "_thrust"] = (
                aircraft_thrust * thrust_distribution[self.propulsor_names.index(propulsor_name)]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        number_of_propulsor = len(self.propulsor_names)

        untouched_thrust_distributor = inputs["data:propulsion:he_power_train:thrust_distribution"]

        # First we force to zero all the propeller that are not connected if we are not using
        # temp copies as is the case for the advanced aerodynamic wing computation
        for propulsor_name in self.propulsor_names:
            if not self.propulsor_connected[propulsor_name] and not self.options[
                "power_train_file_path"
            ].endswith("temp_copy.yml"):
                untouched_thrust_distributor[self.propulsor_names.index(propulsor_name)] = 1e-6
                _LOGGER.warning(
                    propulsor_name
                    + "is not connected to any source. Its contribution to the total aircraft "
                    "thrust was set to zero"
                )

        # Then we rescale the thrust distribution so that at all point it is between 0 and 1
        scale_factor = np.sum(untouched_thrust_distributor)
        thrust_distribution = untouched_thrust_distributor / scale_factor
        aircraft_thrust = inputs["thrust"]

        for propulsor_name in self.propulsor_names:
            if self.propulsor_connected[propulsor_name]:
                partials[propulsor_name + "_thrust", "thrust"] = (
                    np.ones(number_of_points)
                    * thrust_distribution[self.propulsor_names.index(propulsor_name)]
                )

                base_partials = (
                    (-np.tile(aircraft_thrust, (number_of_propulsor, 1)))
                    * thrust_distribution[self.propulsor_names.index(propulsor_name)]
                    / scale_factor
                )
                base_partials[self.propulsor_names.index(propulsor_name), :] = (
                    aircraft_thrust
                    * (1.0 - thrust_distribution[self.propulsor_names.index(propulsor_name)])
                    / scale_factor
                )
                partials[
                    propulsor_name + "_thrust", "data:propulsion:he_power_train:thrust_distribution"
                ] = np.transpose(base_partials)
