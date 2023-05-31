# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_DC_DC_CONVERTER_WEIGHT

oad.RegisterSubmodel.active_models[
    SUBMODEL_DC_DC_CONVERTER_WEIGHT
] = "fastga_he.submodel.propulsion.dc_dc_converter.weight.sum"


@oad.RegisterSubmodel(
    SUBMODEL_DC_DC_CONVERTER_WEIGHT,
    "fastga_he.submodel.propulsion.dc_dc_converter.weight.power_to_mass",
)
class SizingDCDCConverterWeight(om.ExplicitComponent):
    """
    Computation of the weight of the DC/DC converter, based on a simple power to mass for now.
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
            + ":current_in_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the DC/DC converter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the converter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
            units="W/kg",
            val=4000.0,
            desc="Power density of the converter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            units="kg",
            val=20.0,
            desc="Mass of the converter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":power_density"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":power_density"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":power_density"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":power_density"
            ]
            ** 2.0
        )


@oad.RegisterSubmodel(
    SUBMODEL_DC_DC_CONVERTER_WEIGHT,
    "fastga_he.submodel.propulsion.dc_dc_converter.weight.sum",
)
class SizingDCDCConverterWeightBySum(om.Group):
    """
    Computation of the weight of the DC/DC converter, based on the sum of the weight of all
    components.
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

        self.add_subsystem(
            name="weight_by_sum",
            subsys=_SizingDCDCConverterWeightBySum(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_density",
            subsys=_SizingDCDCConverterPowerDensity(dc_dc_converter_id=dc_dc_converter_id),
            promotes=["*"],
        )


class _SizingDCDCConverterWeightBySum(om.ExplicitComponent):
    """
    Computation of the weight of the DC/DC converter, based on the sum of the weight of all
    components.
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
            + ":inductor:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the inductor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:mass",
            val=np.nan,
            units="kg",
            desc="Mass of the capacitor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:mass",
            units="kg",
            val=np.nan,
            desc="Weight of the diode and IGBT module",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":control_card:mass",
            val=1.0,
            units="kg",
            desc="Weight of the control card, is generally constant, taken at 1 kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":contactor:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the contactor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":heat_sink:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the heat sink, includes tubes and core",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            units="kg",
            val=20.0,
            desc="Mass of the converter",
        )

        self.declare_partials(of="*", wrt="*", val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:mass"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:mass"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":module:mass"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":control_card:mass"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":contactor:mass"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":heat_sink:mass"
            ]
        )


class _SizingDCDCConverterPowerDensity(om.ExplicitComponent):
    """
    Computation of the power density of the DC/DC converter, not used, just a figure of merit
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
            + ":current_in_caliber",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
            val=np.nan,
            units="V",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the converter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
            units="kW/kg",
            val=10.0,
            desc="Power density of the DC/DC converter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
            / (
                1000.0
                * inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
                ]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber",
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber"
        ] / (
            1000.0
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_caliber",
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_caliber"
        ] / (
            1000.0
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":power_density",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":current_in_caliber"
            ]
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_in_caliber"
            ]
            / (
                1000.0
                * inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
                ]
                ** 2.0
            )
        )
