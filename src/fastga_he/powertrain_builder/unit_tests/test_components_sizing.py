# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# This test file does not test the validity of the formulas used for the sizing, rather it check
# that the Sizing components outputs what is expect of them i.e: mass, CG, Cd0, ...

from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.powertrain_builder import resources

from fastga_he.models.propulsion.assemblers import sizing_from_pt_file

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    SizingPropeller,
    SizingPMSM,
    SizingInverter,
    SizingDCBus,
    SizingHarness,
    SizingDCDCConverter,
    SizingBatteryPack,
    SizingDCSSPC,
    SizingDCSplitter,
    SizingRectifier,
    SizingGenerator,
    SizingICE,
    SizingFuelTank,
    SizingFuelSystem,
    SizingTurboshaft,
    SizingSpeedReducer,
    SizingPlanetaryGear,
    SizingTurboGenerator,
    SizingGearbox,
    SizingDCAuxLoad,
)

from tests.testing_utilities import VariableListLocal

UNIQUE_STRING = "ca_part_sur_un_depart"


def test_all_sizing_components_exist():

    # Component existing mean that they are import in the right place (the __init__ of the
    # components folder) and that it can be created

    module = __import__("fastga_he.models.propulsion.components", fromlist=[""])

    for component_om_name in resources.DICTIONARY_CN:

        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]

        try:

            klass = getattr(module, sizing_group_name)
            assert klass

        except AttributeError:

            assert False


def test_all_components_output_required_value():

    # Originally I planned on doing mass, cg, Cd0 each on their own but since it takes so much
    # bloody time to list output, we will do everything at once

    for component_om_name in resources.DICTIONARY_CN:

        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]
        sizing_group_id = resources.DICTIONARY_CN_ID[component_om_name]

        klass = globals()[sizing_group_name]
        component = klass()
        # Need a unique string for the rest of the test
        component.options[sizing_group_id] = UNIQUE_STRING

        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
        variables = VariableListLocal.from_system(component)

        output_value = []

        for var in variables:
            if not var.is_input:
                output_value.append(var.name.split(UNIQUE_STRING + ":")[-1])

        assert "mass" in output_value
        assert "CG:x" in output_value
        assert "CG:y" in output_value
        assert "low_speed:CD0" in output_value
        assert "cruise:CD0" in output_value


def test_all_sizing_components_are_imported():

    sizing_assembler_file_path = sizing_from_pt_file.__file__

    r = open(sizing_assembler_file_path, "r")
    lines = r.readlines()

    imported_class = []

    # First we parse the file to check which class are imported and the we check every registered
    # component is imported
    for line in lines:
        if "    Sizing" in line:
            imported_class.append(line.replace("    ", "").replace(",", "").replace("\n", ""))

    for component_om_name in resources.DICTIONARY_CN:
        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]
        assert sizing_group_name in imported_class
