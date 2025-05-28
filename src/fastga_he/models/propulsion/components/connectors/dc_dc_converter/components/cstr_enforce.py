# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_INDUCTOR,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_CAPACITOR,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_MODULE,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_IN,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE_IN,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_FREQUENCY,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_LOSSES,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_POWER_IN,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_CAPACITOR] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.capacitor.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_INDUCTOR] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.inductor.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_MODULE] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.module.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_IN] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.input.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE_IN] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.input.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_LOSSES] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.losses.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_FREQUENCY] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.frequency.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_POWER_IN] = (
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.power.input.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_CAPACITOR,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.capacitor.enforce",
)
class ConstraintsCurrentCapacitorEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the capacitor in the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            val=500.0,
            units="A",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_caliber"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_INDUCTOR,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.inductor.enforce",
)
class ConstraintsCurrentInductorEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the inductor in the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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
            + ":inductor:current_max",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            val=500.0,
            units="A",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_MODULE,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.module.enforce",
)
class ConstraintsCurrentModuleEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the IGBT module in the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            val=500.0,
            units="A",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber"
        ] = max(
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

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

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
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
            ] = 0.0
        else:
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:current_max",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:current_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:current_max",
            ] = 1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_IN,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.input.enforce",
)
class ConstraintsCurrentInputEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen at the input of the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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
            + ":current_in_max",
            val=np.nan,
            units="A",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            val=500.0,
            units="A",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.enforce",
)
class ConstraintsVoltageEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen by the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            val=800.0,
            units="V",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_caliber"
        ] = max(
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max"
            ],
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max"
            ],
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        if (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max"
            ]
            < inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max"
            ]
        ):
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
            ] = 0.0

        else:
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_caliber",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_max",
            ] = 1.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE_IN,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.input.enforce",
)
class ConstraintsVoltageInputEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen at the input of the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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
            + ":voltage_in_max",
            val=np.nan,
            units="V",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            val=500.0,
            units="V",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_LOSSES,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.losses.enforce",
)
class ConstraintsLossesEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum losses seen by the converter during the mission are used
    for the sizing, ensuring a fitted design of each component.
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
            + ":losses_max",
            val=np.nan,
            units="W",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dissipable_heat",
            val=800.0,
            units="W",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dissipable_heat",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":losses_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dissipable_heat"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":losses_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_FREQUENCY,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.frequency.enforce",
)
class ConstraintsFrequencyEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum switching frequency seen by the DC/DC converter
    during the mission is used for the sizing, ensuring a fitted design of each component.
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
            + ":switching_frequency_max",
            units="Hz",
            val=np.nan,
            desc="Maximum switching frequency seen during the mission in the converter",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            units="Hz",
            val=15.0e3,
            desc="Maximum switching frequency of the IGBT module in the converter",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_max"
        ]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_POWER_IN,
    "fastga_he.submodel.propulsion.constraints.dc_dc_converter.power.input.enforce",
)
class ConstraintsPowerInputEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power input seen by the converter during the mission are
    used for the sizing, ensuring a fitted design of each component.
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
            + ":dc_power_in_max",
            val=np.nan,
            units="kW",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_rating",
            val=1.0e3,
            units="kW",
            desc="Power rating of the converter",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_rating",
            wrt="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_rating"
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_max"
        ]
