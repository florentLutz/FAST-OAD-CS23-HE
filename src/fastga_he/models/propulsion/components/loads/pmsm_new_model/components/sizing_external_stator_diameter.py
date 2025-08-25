# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingExtStatorDiameter(om.ExplicitComponent):
    """ Computation of the external stator diameter of a cylindrical PMSM."""

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        # Equation II-43: Stator inner radius R

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter"] = (
            2
            * (
                inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
                + inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height"]
            )
            + inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
        ] = 1

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
        ] = 1

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
        ] = 1
