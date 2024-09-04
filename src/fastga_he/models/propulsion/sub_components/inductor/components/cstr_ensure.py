# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP] = (
    "fastga_he.submodel.propulsion.constraints.inductor.air_gap.ensure"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP,
    "fastga_he.submodel.propulsion.constraints.inductor.air_gap.ensure",
)
class ConstraintsInductorAirGapEnsure(om.ExplicitComponent):
    """
    Class that enforces that computes the difference between the air gap and the maximum value is
    is allowed to take, ensuring it is below its minimum.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            prefix + ":inductor:core_dimension:C",
            units="m",
            val=np.nan,
            desc="C dimension of the E-core in the inductor",
        )
        self.add_input(
            name=prefix + ":inductor:air_gap",
            units="m",
            val=np.nan,
            desc="Air gap in the inductor",
        )

        constraints_prefix = prefix.replace("data", "constraints")
        self.add_output(
            name=constraints_prefix + ":inductor:air_gap",
            units="m",
            val=-1e-3,
            desc="Constraints on maximum air gap in the inductor, respected if <0",
        )

        self.declare_partials(
            of=constraints_prefix + ":inductor:air_gap",
            wrt=[
                prefix + ":inductor:core_dimension:C",
                prefix + ":inductor:air_gap",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]
        constraints_prefix = prefix.replace("data", "constraints")

        outputs[constraints_prefix + ":inductor:air_gap"] = (
            inputs[prefix + ":inductor:air_gap"]
            - 0.1 * inputs[prefix + ":inductor:core_dimension:C"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]
        constraints_prefix = prefix.replace("data", "constraints")

        partials[constraints_prefix + ":inductor:air_gap", prefix + ":inductor:air_gap"] = 1.0
        partials[
            constraints_prefix + ":inductor:air_gap", prefix + ":inductor:core_dimension:C"
        ] = -0.1
