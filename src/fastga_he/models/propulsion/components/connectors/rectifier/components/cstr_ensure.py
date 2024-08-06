# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE,
    SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN,
    SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY,
    SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES,
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE,
    "fastga_he.submodel.propulsion.constraints.rectifier.current.input.rms_one_phase.ensure",
)
class ConstraintsCurrentRMS1PhaseEnsure(om.ExplicitComponent):
    """
    Class that ensure that the maximum RMC current in one phase of the rectifier during the
    mission is below the one used for the sizing, ensuring the component works below its minimum.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
            units="A",
            val=np.nan,
            desc="Maximum RMS current flowing through one arm of the rectifier",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            units="A",
            val=np.nan,
            desc="Caliber RMS current flowing through one arm of the rectifier, used for sizing",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":current_ac_caliber",
            units="A",
            val=-15.0,
            desc="Ensure the 1 phase RMS current rating is enough for the maximum it sees "
            "during the mission, constraint is respected if < 0",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":current_ac_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max"]
            - inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":current_ac_caliber",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":current_ac_caliber",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN,
    "fastga_he.submodel.propulsion.constraints.rectifier.voltage.input.peak.ensure",
)
class ConstraintsVoltagePeakEnsure(om.ExplicitComponent):
    """
    Class that ensure that the maximum peak voltage at the input of the rectifier during the
    mission is below the one used for the sizing, ensuring each component works below its minimum.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max",
            units="V",
            val=np.nan,
            desc="Maximum peak voltage at the input of the rectifier",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
            units="V",
            val=np.nan,
            desc="Caliber peak voltage at the input of the rectifier, used for sizing",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":voltage_ac_caliber",
            units="V",
            val=-80.085,
            desc="Caliber peak voltage at the input of the rectifier, used for sizing",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":voltage_ac_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max"]
            - inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":voltage_ac_caliber",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":voltage_ac_caliber",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY,
    "fastga_he.submodel.propulsion.constraints.rectifier.frequency.ensure",
)
class ConstraintsFrequencyEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum frequency seen by the rectifier during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency_max",
            val=np.nan,
            units="Hz",
            desc="Maximum switching frequency seen by the IGBT modules in the rectifier during the "
            "mission",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            val=np.nan,
            units="Hz",
            desc="Maximum switching frequency of the IGBT modules in the rectifier, used for sizing",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            val=0.0,
            units="Hz",
            desc="Constraints on the maximum switching frequency of the IGBT modules in the "
            "rectifier, respected when <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            wrt=[
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":switching_frequency_max",
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency"
        ] = (
            inputs[
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":switching_frequency_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES,
    "fastga_he.submodel.propulsion.constraints.rectifier.losses.ensure",
)
class ConstraintsLossesEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum losses seen by the rectifier during the
    mission and the dissipable heat used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
            val=np.nan,
            units="W",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            val=np.nan,
            units="W",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":dissipable_heat",
            val=800.0,
            units="W",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":dissipable_heat",
            wrt=[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "constraints:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat"
        ] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max"]
            - inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "constraints:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
        ] = -1.0
