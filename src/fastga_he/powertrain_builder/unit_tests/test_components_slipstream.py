# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# This test file does not test the validity of the formulas used for the sizing, rather it check
# that the Sizing components outputs what is expect of them i.e: mass, CG, Cd0, ...

from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.powertrain_builder import resources

from fastga_he.models.propulsion.assemblers import delta_from_pt_file

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    SlipstreamPropeller,
    SlipstreamPMSM,
    SlipstreamInverter,
    SlipstreamDCBus,
    SlipstreamHarness,
    SlipstreamDCDCConverter,
    SlipstreamBatteryPack,
    SlipstreamDCSSPC,
    SlipstreamDCSplitter,
    SlipstreamRectifier,
    SlipstreamGenerator,
    SlipstreamICE,
    SlipstreamFuelTank,
    SlipstreamFuelSystem,
    SlipstreamTurboshaft,
    SlipstreamSpeedReducer,
    SlipstreamPlanetaryGear,
    SlipstreamTurboGenerator,
    SlipstreamGearbox,
)

from tests.testing_utilities import VariableListLocal

UNIQUE_STRING = "ca_part_sur_un_depart"


def test_all_slipstream_components_exist():

    # Component existing mean that they are imported in the right place (the __init__ of the
    # components folder) and that it can be created

    module = __import__("fastga_he.models.propulsion.components", fromlist=[""])

    for component_om_name in resources.DICTIONARY_CN:

        slipstream_group_name = "Slipstream" + resources.DICTIONARY_CN[component_om_name]

        try:

            klass = getattr(module, slipstream_group_name)
            assert klass

        except AttributeError:

            assert False


def test_all_components_output_required_value():

    # Originally I planned on doing each delta on their own but since it takes so much bloody
    # time to list output, we will do everything at once

    for component_om_name in resources.DICTIONARY_CN:

        slipstream_group_name = "Slipstream" + resources.DICTIONARY_CN[component_om_name]
        slipstream_group_id = resources.DICTIONARY_CN_ID[component_om_name]

        klass = globals()[slipstream_group_name]
        component = klass()
        # Need a unique string for the rest of the test
        component.options[slipstream_group_id] = UNIQUE_STRING

        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
        variables = VariableListLocal.from_system(component)

        input_and_output_value = []

        for var in variables:
            input_and_output_value.append(var.name)

        # Because there are a lot of IVC which makes the delta's appear as inputs
        assert "delta_Cl" in input_and_output_value
        assert "delta_Cd" in input_and_output_value
        assert "delta_Cm" in input_and_output_value


def test_all_sizing_components_are_imported():

    slipstream_assembler_file_path = delta_from_pt_file.__file__

    r = open(slipstream_assembler_file_path, "r")
    lines = r.readlines()

    imported_class = []

    # First we parse the file to check which class are imported and the we check every registered
    # component is imported
    for line in lines:
        if "    Slipstream" in line:
            imported_class.append(line.replace("    ", "").replace(",", "").replace("\n", ""))

    for component_om_name in resources.DICTIONARY_CN:
        slipstream_group_name = "Slipstream" + resources.DICTIONARY_CN[component_om_name]
        assert slipstream_group_name in imported_class
