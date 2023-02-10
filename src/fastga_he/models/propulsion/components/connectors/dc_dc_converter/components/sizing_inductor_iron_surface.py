# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorIronSurface(om.ExplicitComponent):
    """
    Computation of the iron surface of the filter inductor necessary to store the energy,
    implementation  in the openMDAO format of :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_energy_rating",
            units="J",
            val=np.nan,
            desc="Maximum magnetic energy that can be stored in the inductors",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap",
            units="m",
            val=np.nan,
            desc="Air gap in the inductor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_field",
            units="T",
            val=1.0,
            desc="Magnetic field in the inductor core, will be taken at its maximum allowable "
            "value to reduce the weight",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability",
            units="H/m",
            val=14 * 3.14e-7,
            desc="Permeability of the iron core",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            units="m**2",
            val=700e-6,
            desc="Iron surface of the E-core inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        energy_rating = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_energy_rating"
        ]
        air_gap = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap"
        ]
        magnetic_field = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_field"
        ]
        mu = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability"
        ]

        iron_area = energy_rating * 2.0 * mu / (magnetic_field ** 2.0 * 2 * air_gap)

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface"
        ] = iron_area

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        energy_rating = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_energy_rating"
        ]
        air_gap = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap"
        ]
        magnetic_field = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_field"
        ]
        mu = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_energy_rating",
        ] = (
            2.0 * mu / (magnetic_field ** 2.0 * 2 * air_gap)
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap",
        ] = (
            -energy_rating * 2.0 * mu / (magnetic_field ** 2.0 * 2 * air_gap ** 2.0)
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:magnetic_field",
        ] = (
            -4.0 * energy_rating * mu / (magnetic_field ** 3.0 * 2 * air_gap)
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability",
        ] = (
            energy_rating * 2.0 / (magnetic_field ** 2.0 * 2 * air_gap)
        )
