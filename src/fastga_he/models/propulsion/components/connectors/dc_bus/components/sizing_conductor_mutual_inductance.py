# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarMutualInductance(om.ExplicitComponent):
    """
    Computation of the conductor plates mutual inductance.

    Based on the formula from :cite:`khan:2014`.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
            units="m",
            val=0.3,
            desc="Length of the bus bar conductor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity",
            val=4,
            desc="Dielectric permittivity of the insulation, chosen as Gexol insulation",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:"
            + dc_bus_id
            + ":conductor:mutual_inductance",
            units="H",
            val=2e-7,
            desc="Self inductance of the bus bar conductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length"
        ]
        # The formula actually requires the distance between the plates and not the insulation
        # thickness but here the choice was made to take them as equal
        insulation_thickness = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"
        ]
        mu_r = inputs[
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity"
        ]

        # mu_r in the fromula was taken as permeability not relative permeability
        mutual_inductance = (
            2.0e-7
            * mu_r
            * (
                conductor_length
                * np.log(
                    (
                        np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0)
                        + conductor_length
                    )
                    / insulation_thickness
                )
                - np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0)
                + insulation_thickness
            )
        )

        outputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:mutual_inductance"
        ] = mutual_inductance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_bus_id = self.options["dc_bus_id"]

        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length"
        ]
        insulation_thickness = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"
        ]
        mu_r = inputs[
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity"
        ]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:mutual_inductance",
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity",
        ] = 2.0e-7 * (
            conductor_length
            * np.log(
                (np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0) + conductor_length)
                / insulation_thickness
            )
            - np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0)
            + insulation_thickness
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:mutual_inductance",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
        ] = (
            2.0e-7
            * mu_r
            * np.log(
                (np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0) + conductor_length)
                / insulation_thickness
            )
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:mutual_inductance",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
        ] = (
            2.0e-7
            * mu_r
            * (
                -np.sqrt(conductor_length ** 2.0 + insulation_thickness ** 2.0)
                - insulation_thickness
            )
            / insulation_thickness
        )
