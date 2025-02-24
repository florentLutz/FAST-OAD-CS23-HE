# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

DEFAULT_FC_POWER_DENSITY = 124  # [kW/m^3]
DEFAULT_FC_VOLUME = 1.62e-3  # [m^3]


class SizingPEMFCStackVolume(om.ExplicitComponent):
    """
    Computation of the volume the PEMFC based on number of layers. The calculation is based on the
    equations given by :cite:`hoogendoorn:2018`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            units="kW/m**3",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            units="m**3",
            val=5.0,
            desc="Volume of PEMFC stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume"] = (
            DEFAULT_FC_VOLUME
            * DEFAULT_FC_POWER_DENSITY
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
        ] = (
            -DEFAULT_FC_VOLUME
            * DEFAULT_FC_POWER_DENSITY
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
            ]
            ** 2
        )
