# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class InverterEnergyCoefficients(om.ExplicitComponent):
    """Computation of coefficients of the computation of commutation energy in diodes and IGBT.
    Reference for IGBT and diode is the SEMiX703GB126HDs from Semikron."""

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
            default=1.36e-9,
            desc="Reference coefficient in front of current square for ON commutation energy",
        )
        self.options.declare(
            name="a_off_ref",
            types=float,
            default=-9.624e-11,
            desc="Reference coefficient in front of current square for OFF commutation energy",
        )
        self.options.declare(
            name="a_rr_ref",
            types=float,
            default=-8.285e-10,
            desc="Reference coefficient in front of current square for RR commutation energy",
        )

        self.options.declare(
            name="b_on_ref",
            types=float,
            default=3.986e-6,
            desc="Reference coefficient in front of current for ON commutation energy",
        )
        self.options.declare(
            name="b_off_ref",
            types=float,
            default=1.330e-5,
            desc="Reference coefficient in front of current for OFF commutation energy",
        )
        self.options.declare(
            name="b_rr_ref",
            types=float,
            default=1.116e-5,
            desc="Reference coefficient in front of current for RR commutation energy",
        )

        self.options.declare(
            name="c_on_ref",
            types=float,
            default=1.214e-3,
            desc="Reference constant coefficient for ON commutation energy",
        )
        self.options.declare(
            name="c_off_ref",
            types=float,
            default=9.064e-4,
            desc="Reference constant coefficient for OFF commutation energy",
        )
        self.options.declare(
            name="c_rr_ref",
            types=float,
            default=1.249e-3,
            desc="Reference constant coefficient for RR commutation energy",
        )

    def setup(self):

        inverted_id = self.options["inverter_id"]

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a", val=np.nan
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c", val=np.nan
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:c",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:c",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:a",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:b",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:c",
        )

        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:a",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:a",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:a",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:b",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:b",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:b",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:c",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:c",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:c",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverted_id = self.options["inverter_id"]

        a_star = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a"]
        c_star = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c"]

        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:a"] = (
            a_star * self.options["a_on_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:a"] = (
            a_star * self.options["a_rr_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:a"] = (
            a_star * self.options["a_off_ref"]
        )

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:b"
        ] = self.options["b_on_ref"]
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:b"
        ] = self.options["b_rr_ref"]
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:b"
        ] = self.options["b_off_ref"]

        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:c"] = (
            c_star * self.options["c_on_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:c"] = (
            c_star * self.options["c_rr_ref"]
        )
        outputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:c"] = (
            c_star * self.options["c_off_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverted_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:a",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = self.options["a_on_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:a",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = self.options["a_rr_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:a",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = self.options["a_off_ref"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:b",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = 0.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:b",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = 0.0
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:b",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:a",
        ] = 0.0

        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_on:c",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c",
        ] = self.options["c_on_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_rr:c",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c",
        ] = self.options["c_rr_ref"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":energy_off:c",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":scaling:c",
        ] = self.options["c_off_ref"]
