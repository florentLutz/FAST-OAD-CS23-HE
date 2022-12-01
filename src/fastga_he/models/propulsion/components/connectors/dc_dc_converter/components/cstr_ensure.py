# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.ensure",
)
class ConstraintsEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum seen by the DC/DC converter during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            val=0.0,
            units="A",
            desc="Respected if negative",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:current_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:current_caliber",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            val=0.0,
            units="A",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_caliber",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:current_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:current_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            val=0.0,
            units="A",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            val=0.0,
            units="A",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max",
            val=np.nan,
            units="V",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
            val=np.nan,
            units="V",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            val=np.nan,
            units="V",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            val=0.0,
            units="V",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
            ],
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            val=np.nan,
            units="V",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            val=0.0,
            units="V",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber",
            ],
            method="exact",
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_max",
            units="Hz",
            val=np.nan,
            desc="Maximum switching frequency seen during the mission in the converter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            units="Hz",
            val=np.nan,
            desc="Maximum switching frequency of the IGBT module in the converter",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            units="Hz",
            val=0.0,
            desc="Constraints on maximum switching frequency of the IGBT module in the converter, "
            "respected if <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:current_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:current_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber"
        ] = (
            max(
                inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":diode:current_max"
                ],
                inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":igbt:current_max"
                ],
            )
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber"
        ] = (
            max(
                inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":voltage_out_max"
                ],
                inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":voltage_in_max"
                ],
            )
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
        )

        outputs[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
        ] = -1.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
        ] = -1.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
        ] = -1.0
        if (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max"
            ]
            < inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max"
            ]
        ):

            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
            ] = 0.0
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
            ] = 1.0
        else:
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
            ] = 1.0
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
            ] = 0.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
        ] = -1.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
        ] = -1.0
        if (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max"
            ]
            < inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max"
            ]
        ):

            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
            ] = 0.0
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
            ] = 1.0
        else:
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
            ] = 1.0
            partials[
                "constraints:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
            ] = 0.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
        ] = -1.0

        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
        ] = -1.0
