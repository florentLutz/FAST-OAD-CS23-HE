# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# This test file does not test the validity of the formulas used for the performances, rather it
# check that the Performances components outputs what is expect of them i.e: sources must output
# fuel_consumed_t, energy_consumed_t, propulsive load must output power rate, , propulsor must
# have thrust as an input, ...

from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.powertrain_builder import resources

from fastga_he.models.propulsion.assemblers import performances_from_pt_file

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    PerformancesPropeller,
    PerformancesPMSM,
    PerformancesInverter,
    PerformancesDCBus,
    PerformancesHarness,
    PerformancesDCDCConverter,
    PerformancesBatteryPack,
    PerformancesDCSSPC,
    PerformancesDCSplitter,
    PerformancesRectifier,
    PerformancesGenerator,
    PerformancesICE,
    PerformancesFuelTank,
    PerformancesFuelSystem,
    PerformancesTurboshaft,
    PerformancesSpeedReducer,
    PerformancesPlanetaryGear,
)

from tests.testing_utilities import VariableListLocal

UNIQUE_STRING = "ca_part_sur_un_depart"


def test_all_performances_components_exist():

    # Component existing mean that they are import in the right place (the __init__ of the
    # components folder) and that it can be created

    module = __import__("fastga_he.models.propulsion.components", fromlist=[""])

    for component_om_name in resources.DICTIONARY_CN:

        performances_group_name = "Performances" + resources.DICTIONARY_CN[component_om_name]

        try:

            klass = getattr(module, performances_group_name)
            assert klass

        except AttributeError:

            assert False


def test_all_performances_components_are_imported():

    performances_assembler_file_path = performances_from_pt_file.__file__

    r = open(performances_assembler_file_path, "r")
    lines = r.readlines()

    imported_class = []

    # First we parse the file to check which class are imported and the we check every registered
    # component is imported
    for line in lines:
        if "    Performances" in line:
            imported_class.append(line.replace("    ", "").replace(",", "").replace("\n", ""))

    for component_om_name in resources.DICTIONARY_CN:
        performances_group_name = "Performances" + resources.DICTIONARY_CN[component_om_name]
        assert performances_group_name in imported_class


def test_all_components_output_required_value():

    # Originally I planned on each type of components on their own but since it takes so much
    # bloody time to list outputs and inputs, we will do everything at once

    for component_om_name in resources.DICTIONARY_CN:

        performances_group_name = "Performances" + resources.DICTIONARY_CN[component_om_name]
        performances_group_id = resources.DICTIONARY_CN_ID[component_om_name]
        component_type = resources.DICTIONARY_CTC[component_om_name]

        klass = globals()[performances_group_name]
        component = klass()
        # Need a unique string for the rest of the test
        component.options[performances_group_id] = UNIQUE_STRING

        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
        variables = VariableListLocal.from_system(component)

        if "propulsor" in component_type:

            input_value = []

            for var in variables:
                if var.is_input:
                    input_value.append(var.name)

            assert "thrust" in input_value

        if "propulsive_load" in component_type:

            output_value = []

            for var in variables:
                if not var.is_input:
                    output_value.append(var.name)

            assert "shaft_power_for_power_rate" in output_value

        if "source" in component_type:

            output_value = []
            input_value = []

            for var in variables:
                if not var.is_input:
                    output_value.append(var.name)
                else:
                    input_value.append(var.name)

            # If it's in an ivc (like fuel for batteries), it will be listed in the inputs
            assert "fuel_consumed_t" in output_value or "fuel_consumed_t" in input_value
            assert (
                "non_consumable_energy_t" in output_value
                or "non_consumable_energy_t" in input_value
            )

        if "tank" in component_type:

            output_value = []

            for var in variables:
                if not var.is_input:
                    output_value.append(var.name)

            assert "fuel_remaining_t" in output_value
