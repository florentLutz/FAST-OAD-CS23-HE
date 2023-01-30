# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterHeatSinkTubeWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tube running in the heat sink based on the geometry and
    density. Method from :cite:`giraud:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length",
            units="m",
            val=np.nan,
            desc="Length of the heat sink",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":tube:number_of_passes",
            val=4,
            desc="Number of passes in the heat sink (between 2 and 6 usually)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:density",
            units="kg/m**3",
            val=8960.0,
            desc="Density of the tube for the cooling of the inverter, by default copper is "
            "assumed",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            units="kg",
            val=0.10,
            desc="Wieght of the tube for the cooling of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        length_hs = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length"
        ]
        number_of_passes = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes"
        ]
        inner_diameter = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter"
        ]
        outer_diameter = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter"
        ]
        tube_density = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:density"
        ]

        # Tube length outside of the cold plat assumed at 10% of the total length
        outside_length = 0.1 * length_hs

        tube_weight = (
            (
                2.0 * outside_length
                + number_of_passes * length_hs
                + outer_diameter * np.pi * (number_of_passes - 1.0)
            )
            * (np.pi / 4 * (outer_diameter ** 2.0 - inner_diameter ** 2.0))
            * tube_density
        )

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass"
        ] = tube_weight

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        length_hs = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length"
        ]
        number_of_passes = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes"
        ]
        inner_diameter = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter"
        ]
        outer_diameter = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter"
        ]
        tube_density = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:density"
        ]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:length",
        ] = (
            (np.pi / 4 * (outer_diameter ** 2.0 - inner_diameter ** 2.0))
            * tube_density
            * (0.2 + number_of_passes)
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":tube:number_of_passes",
        ] = (
            (length_hs + outer_diameter * np.pi)
            * (np.pi / 4 * (outer_diameter ** 2.0 - inner_diameter ** 2.0))
            * tube_density
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter",
        ] = -(
            (
                2.0 * 0.1 * length_hs
                + number_of_passes * length_hs
                + outer_diameter * np.pi * (number_of_passes - 1.0)
            )
            * (np.pi / 2 * inner_diameter)
            * tube_density
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
        ] = tube_density * (
            (np.pi * (number_of_passes - 1.0))
            * (np.pi / 4 * (outer_diameter ** 2.0 - inner_diameter ** 2.0))
            + (
                2.0 * 0.1 * length_hs
                + number_of_passes * length_hs
                + outer_diameter * np.pi * (number_of_passes - 1.0)
            )
            * (np.pi / 2 * outer_diameter)
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:density",
        ] = (
            2.0 * 0.1 * length_hs
            + number_of_passes * length_hs
            + outer_diameter * np.pi * (number_of_passes - 1.0)
        ) * (
            np.pi / 4 * (outer_diameter ** 2.0 - inner_diameter ** 2.0)
        )
