# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorReluctance(om.ExplicitComponent):
    """
    Computation of the reluctance of the filter inductor, implementation of the formula from
    :cite:`budinger_sizing_2023`.
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
            + ":inductor:air_gap",
            units="m",
            val=np.nan,
            desc="Air gap in the inductor",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability",
            units="H/m",
            val=14 * 3.14e-7,
            desc="Permeability of the iron core",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            units="m**2",
            val=np.nan,
            desc="Iron surface of the E-core inductor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:reluctance",
            units="H**-1",
            val=2.00e9,
            desc="Reluctance of the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        mu = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability"
        ]
        air_gap = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap"
        ]
        iron_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface"
        ]

        reluctance = 2.0 * air_gap / (mu * iron_area)

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:reluctance"
        ] = reluctance

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        mu = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability"
        ]
        air_gap = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap"
        ]
        iron_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:reluctance",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_permeability",
        ] = (-2.0 * air_gap / (mu ** 2.0 * iron_area))
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:reluctance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:air_gap",
        ] = 2.0 / (mu * iron_area)
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:reluctance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
        ] = (-2.0 * air_gap / (mu * iron_area ** 2.0))
