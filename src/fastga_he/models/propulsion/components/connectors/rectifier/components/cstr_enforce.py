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

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE] = (
    "fastga_he.submodel.propulsion.constraints.rectifier.current.input.rms_one_phase.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN] = (
    "fastga_he.submodel.propulsion.constraints.rectifier.voltage.input.peak.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY] = (
    "fastga_he.submodel.propulsion.constraints.rectifier.frequency.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES] = (
    "fastga_he.submodel.propulsion.constraints.rectifier.losses.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_CURRENT_IN_RMS_1_PHASE,
    "fastga_he.submodel.propulsion.constraints.rectifier.current.input.rms_one_phase.enforce",
)
class ConstraintsCurrentRMS1PhaseEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum RMC current in one phase of the rectifier during the
    mission is used for the sizing, ensuring a fitted design of each component.
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
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            units="A",
            val=200.0,
            desc="Caliber RMS current flowing through one arm of the rectifier, used for sizing",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ] = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max"]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_VOLTAGE_IN,
    "fastga_he.submodel.propulsion.constraints.rectifier.voltage.input.peak.enforce",
)
class ConstraintsVoltagePeakEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum peak voltage at the input of the rectifier during the
    mission is used for the sizing, ensuring a fitted design of each component.
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
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
            units="V",
            val=800.0,
            desc="Caliber peak voltage at the input of the rectifier, used for sizing",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ] = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max"]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_FREQUENCY,
    "fastga_he.submodel.propulsion.constraints.rectifier.frequency.enforce",
)
class ConstraintsFrequencyEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum frequency seen by the rectifier during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            val=12.0e3,
            units="Hz",
            desc="Maximum switching frequency of the IGBT modules in the rectifier, used for sizing",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency",
            wrt="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency"
        ] = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_LOSSES,
    "fastga_he.submodel.propulsion.constraints.rectifier.losses.enforce",
)
class ConstraintsLossesEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum losses seen by the rectifier during the mission are used
    for the sizing, ensuring a fitted design of each component.
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
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            val=800.0,
            units="W",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":dissipable_heat"] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max"]
        )
