# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_INVERTER_CURRENT,
    SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE,
    SUBMODEL_CONSTRAINTS_INVERTER_LOSSES,
    SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY,
    SUBMODEL_CONSTRAINTS_INVERTER_POWER_OUT,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INVERTER_CURRENT] = (
    "fastga_he.submodel.propulsion.constraints.inverter.current.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE] = (
    "fastga_he.submodel.propulsion.constraints.inverter.voltage.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INVERTER_LOSSES] = (
    "fastga_he.submodel.propulsion.constraints.inverter.losses.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY] = (
    "fastga_he.submodel.propulsion.constraints.inverter.frequency.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_INVERTER_POWER_OUT] = (
    "fastga_he.submodel.propulsion.constraints.inverter.power.output.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_CURRENT,
    "fastga_he.submodel.propulsion.constraints.inverter.current.enforce",
)
class ConstraintsCurrentEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the inverter during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
            val=np.nan,
            units="A",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            val=200.0,
            units="A",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.inverter.voltage.enforce",
)
class ConstraintsVoltageEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen by the inverter during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
            val=np.nan,
            units="V",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            val=800.0,
            units="V",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_LOSSES,
    "fastga_he.submodel.propulsion.constraints.inverter.losses.enforce",
)
class ConstraintsLossesEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum losses seen by the inverter during the mission are used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
            val=np.nan,
            units="W",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            val=800.0,
            units="W",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY,
    "fastga_he.submodel.propulsion.constraints.inverter.frequency.enforce",
)
class ConstraintsFrequencyEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum frequency seen by the inverter during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency_max",
            val=np.nan,
            units="Hz",
            desc="Maximum switching frequency seen by the IGBT modules in the inverter during the "
            "mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
            val=12.0e3,
            units="Hz",
            desc="Maximum switching frequency of the IGBT modules in the inverter, used for sizing",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
            wrt="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency"
        ] = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_POWER_OUT,
    "fastga_he.submodel.propulsion.constraints.inverter.power.output.enforce",
)
class ConstraintsPowerOutputEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power output seen by the inverter during the mission is
    used for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_max",
            val=np.nan,
            units="kW",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_rating",
            val=200.0,
            units="kW",
            desc="Power rating of the inverter",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_rating",
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_rating"
        ] = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":ac_power_out_max"]
