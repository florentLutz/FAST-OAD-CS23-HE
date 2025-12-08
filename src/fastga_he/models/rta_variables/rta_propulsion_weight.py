# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator


class RTAPropulsionWeight(om.ExplicitComponent):
    """
    This component adapts propulsion mass variables for CG computation from FAST-GA-HE to RTA. To
    prevent variable conflicts caused by shared outputs in the powertrain mass computation within
    RTA, the `PowerTrainMassRTA` component under `mass_breakdown` must be added to the configuration
    file to replace the original computation. To ensure the correct CG of the powertrain in the
    aircraft CG calculation, the propeller mass is set to represent the additional weight from the
    original engine. Meanwhile, the engine weight is kept unchanged to ensure overall weight
    consistency for other cabin systems.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])

        self.add_input("data:propulsion:he_power_train:CG:x", val=np.nan, units="m")
        self.add_input("data:propulsion:he_power_train:mass", val=np.nan, units="kg")

        (
            self.propulsive_load_names,
            self.propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        if any(
            component
            in self.configurator._cache[self.configurator._power_train_file]["components_type"]
            for component in ["battery_pack", "PEMFC_stack"]
        ):
            self.add_input(
                "data:propulsion:he_power_train:base_model_engine_mass", val=np.nan, units="kg"
            )

        else:
            for component_type, component_name in zip(
                self.propulsive_load_types, self.propulsive_load_names
            ):
                if component_type != "PMSM" or "SM_PMSM":
                    self.add_input(
                        "data:propulsion:he_power_train:"
                        + component_type
                        + ":"
                        + component_name
                        + ":mass",
                        val=np.nan,
                        units="kg",
                    )

        self.add_output("data:weight:propulsion:propeller:mass", val=520.0, units="kg")
        self.add_output("data:weight:propulsion:engine:mass", val=780.0, units="kg")
        self.add_output("data:weight:propulsion:fuel_lines:mass", val=0.0, units="kg")
        self.add_output(
            "data:weight:propulsion:engine_controls_instrumentation:mass", val=0.0, units="kg"
        )
        self.add_output("data:weight:propulsion:engine:CG:x", units="m", val=10.0)
        self.add_output("data:weight:propulsion:propeller:CG:x", units="m", val=10.0)

    def setup_partials(self):
        self.declare_partials(
            ["data:weight:propulsion:engine:CG:x", "data:weight:propulsion:propeller:CG:x"],
            "data:propulsion:he_power_train:CG:x",
        )
        self.declare_partials(
            "data:weight:propulsion:propeller:mass", "data:propulsion:he_power_train:mass", val=1.0
        )

        if any(
            component
            in self.configurator._cache[self.configurator._power_train_file]["components_type"]
            for component in ["battery_pack", "PEMFC_stack"]
        ):
            self.declare_partials(
                "data:weight:propulsion:engine:mass",
                "data:propulsion:he_power_train:base_model_engine_mass",
                val=1.0,
            )
            self.declare_partials(
                "data:weight:propulsion:propeller:mass",
                "data:propulsion:he_power_train:base_model_engine_mass",
                val=-1.0,
            )
        else:
            for component_type, component_name in zip(
                self.propulsive_load_types, self.propulsive_load_names
            ):
                if component_type != "PMSM" or "SM_PMSM":
                    self.declare_partials(
                        "data:weight:propulsion:engine:mass",
                        "data:propulsion:he_power_train:"
                        + component_type
                        + ":"
                        + component_name
                        + ":mass",
                        val=1.0,
                    )
                    self.declare_partials(
                        "data:weight:propulsion:propeller:mass",
                        "data:propulsion:he_power_train:"
                        + component_type
                        + ":"
                        + component_name
                        + ":mass",
                        val=-1.0,
                    )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        powertrain_cg = inputs["data:propulsion:he_power_train:CG:x"]
        powertrain_weight = inputs["data:propulsion:he_power_train:mass"]
        engine_weight = 0.0

        if any(
            component
            in self.configurator._cache[self.configurator._power_train_file]["components_type"]
            for component in ["battery_pack", "PEMFC_stack"]
        ):
            engine_weight = inputs["data:propulsion:he_power_train:base_model_engine_mass"]

        else:
            for component_type, component_name in zip(
                self.propulsive_load_types, self.propulsive_load_names
            ):
                if component_type != "PMSM" or "SM_PMSM":
                    engine_weight += inputs[
                        "data:propulsion:he_power_train:"
                        + component_type
                        + ":"
                        + component_name
                        + ":mass"
                    ]

        outputs["data:weight:propulsion:engine:mass"] = engine_weight
        outputs["data:weight:propulsion:propeller:mass"] = powertrain_weight - engine_weight
        outputs["data:weight:propulsion:engine:CG:x"] = powertrain_cg
        outputs["data:weight:propulsion:propeller:CG:x"] = powertrain_cg
