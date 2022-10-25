# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingInverterEnergyCoefficients(om.ExplicitComponent):
    """Computation of coefficients of the computation of commutation energy in diodes and IGBT.
    Reference for IGBT and diode is the SEMiX453GB12M7p from Semikron."""

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

        self.options.declare(
            name="a_on_ref",
            types=float,
            default=0.015862856846788793,
            desc="Reference constant coefficient for ON commutation energy",
        )
        self.options.declare(
            name="a_off_ref",
            types=float,
            default=0.01507362815506554,
            desc="Reference constant coefficient for OFF commutation energy",
        )
        self.options.declare(
            name="a_rr_ref",
            types=float,
            default=0.004248159711027629,
            desc="Reference constant coefficient for RR commutation energy",
        )

        self.options.declare(
            name="b_on_ref",
            types=float,
            default=3.3256504177779456e-05,
            desc="Reference coefficient in front of current for ON commutation energy",
        )
        self.options.declare(
            name="b_off_ref",
            types=float,
            default=0.0002539092853205021,
            desc="Reference coefficient in front of current for OFF commutation energy",
        )
        self.options.declare(
            name="b_rr_ref",
            types=float,
            default=0.00034030311012866587,
            desc="Reference coefficient in front of current for RR commutation energy",
        )

        self.options.declare(
            name="c_on_ref",
            types=float,
            default=5.134677955597825e-07,
            desc="Reference coefficient in front of current square for ON commutation energy",
        )
        self.options.declare(
            name="c_off_ref",
            types=float,
            default=-1.739057882676467e-07,
            desc="Reference coefficient in front of current square for OFF commutation energy",
        )
        self.options.declare(
            name="c_rr_ref",
            types=float,
            default=-4.5116538387667774e-08,
            desc="Reference coefficient in front of current square for RR commutation energy",
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a", val=np.nan
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c", val=np.nan
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
        )

        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        a_star = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a"]
        c_star = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a"] = (
            a_star * self.options["a_on_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a"] = (
            a_star * self.options["a_rr_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a"] = (
            a_star * self.options["a_off_ref"]
        )

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b"
        ] = self.options["b_on_ref"]
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b"
        ] = self.options["b_rr_ref"]
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b"
        ] = self.options["b_off_ref"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c"] = (
            c_star * self.options["c_on_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c"] = (
            c_star * self.options["c_rr_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c"] = (
            c_star * self.options["c_off_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = self.options["a_on_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = self.options["a_rr_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = self.options["a_off_ref"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = 0.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = 0.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:a",
        ] = 0.0

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c",
        ] = self.options["c_on_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c",
        ] = self.options["c_rr_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:c",
        ] = self.options["c_off_ref"]
