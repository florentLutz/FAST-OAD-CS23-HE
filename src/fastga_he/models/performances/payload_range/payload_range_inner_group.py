# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .payload_range_inner_sampling import ComputePayloadRangeInnerSampling
from .payload_range_inner import ComputePayloadRangeInner


class ComputePayloadRangeInnerGroup(om.Group):
    """
    Computation of the performances of the aircraft on some points inside the payload range diagram.
    """

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            name="generate_sample",
            types=bool,
            default=False,
            desc="Option to trigger the generation of a sampling of the payload range or "
            "use an existing one",
        )
        self.options.declare(
            name="number_of_sample",
            types=int,
            default=12,
            desc="Number of sample inside the payload range envelope",
        )

    def setup(self):

        if self.options["generate_sample"]:
            self.add_subsystem(
                name="generate_sample",
                subsys=ComputePayloadRangeInnerSampling(
                    number_of_sample=self.options["number_of_sample"]
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            name="compute_payload_range_inner",
            subsys=ComputePayloadRangeInner(
                number_of_sample=self.options["number_of_sample"],
                power_train_file_path=self.options["power_train_file_path"],
            ),
            promotes=["*"],
        )
