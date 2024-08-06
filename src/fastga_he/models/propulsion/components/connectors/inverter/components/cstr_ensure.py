# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_INVERTER_CURRENT,
    SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE,
    SUBMODEL_CONSTRAINTS_INVERTER_LOSSES,
    SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_CURRENT,
    "fastga_he.submodel.propulsion.constraints.inverter.current.ensure",
)
class ConstraintsCurrentEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum current seen by the inverter during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            val=np.nan,
            units="A",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":current_caliber",
            val=0.0,
            units="A",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max"]
            - inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.inverter.voltage.ensure",
)
class ConstraintsVoltageEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum voltage seen by the inverter during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            val=np.nan,
            units="V",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":voltage_caliber",
            val=0.0,
            units="V",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max"]
            - inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_LOSSES,
    "fastga_he.submodel.propulsion.constraints.inverter.losses.ensure",
)
class ConstraintsLossesEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum losses seen by the inverter during the
    mission and the dissipable heat used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            val=np.nan,
            units="W",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":dissipable_heat",
            val=800.0,
            units="W",
            desc="Respected if negative",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max"]
            - inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":dissipable_heat",
        ] = -1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INVERTER_FREQUENCY,
    "fastga_he.submodel.propulsion.constraints.inverter.frequency.ensure",
)
class ConstraintsFrequencyEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum frequency seen by the inverter during the
    mission and the value used for sizing, ensuring each component works below its maxima.
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
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
            val=np.nan,
            units="Hz",
            desc="Maximum switching frequency of the IGBT modules in the inverter, used for sizing",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency",
            val=0.0,
            units="Hz",
            desc="Constraints on the maximum switching frequency of the IGBT modules in the "
            "inverter, respected when <0",
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency",
            wrt=[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_max",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "constraints:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency"
        ] = (
            inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":switching_frequency_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":switching_frequency",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
        ] = -1.0
