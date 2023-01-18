# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_INDUCTOR,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_CAPACITOR,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_MODULE,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_IN,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE_IN,
    SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_FREQUENCY,
)


class ConstraintsDCDCConverter(om.Group):
    """
    Class that gather the different constraints for the DC/DC converter be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        option_dc_dc_converter_id = {"dc_dc_converter_id": self.options["dc_dc_converter_id"]}

        self.add_subsystem(
            name="constraints_capacitor_current",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_CAPACITOR,
                options=option_dc_dc_converter_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_inductor_current",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_INDUCTOR,
                options=option_dc_dc_converter_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_module_current",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_MODULE,
                options=option_dc_dc_converter_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_input_current",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_CURRENT_IN, options=option_dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_voltage",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE, options=option_dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_input_voltage",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_VOLTAGE_IN, options=option_dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="constraints_frequency",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_DC_DC_CONVERTER_FREQUENCY, options=option_dc_dc_converter_id
            ),
            promotes=["*"],
        )
