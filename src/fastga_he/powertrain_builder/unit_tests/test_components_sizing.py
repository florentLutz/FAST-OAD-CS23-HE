# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# This test file does not test the validity of the formulas used for the sizing, rather it check
# that the Sizing components outputs what is expect of them i.e: mass, CG, Cd0, ...

from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.powertrain_builder import resources

from fastga_he.models.propulsion.assemblers import sizing_from_pt_file
import fastga_he.models.propulsion.components as he_comp

from tests.testing_utilities import VariableListLocal

UNIQUE_STRING = "ca_part_sur_un_depart"


def test_all_sizing_components_exist():
    # Component existing mean that they are import in the right place (the __init__ of the
    # components folder) and that it can be created

    for component_om_name in resources.DICTIONARY_CN:
        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]

        try:
            class_to_test = he_comp.__dict__[sizing_group_name]()
            assert class_to_test

        except AttributeError:
            assert False


def test_all_components_output_required_value():
    # Originally I planned on doing mass, cg, Cd0 each on their own but since it takes so much
    # bloody time to list output, we will do everything at once

    for component_om_name in resources.DICTIONARY_CN:
        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]
        sizing_group_id = resources.DICTIONARY_CN_ID[component_om_name]

        component = he_comp.__dict__[sizing_group_name]()
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
    imported_components = list(he_comp.__dict__.keys())

    for component_om_name in resources.DICTIONARY_CN:
        sizing_group_name = "Sizing" + resources.DICTIONARY_CN[component_om_name]
        assert sizing_group_name in imported_components


def test_all_imported_sizing_components_are_defined():
    # In practice this covers the tests above
    imported_components = list(he_comp.__dict__.keys())
    imported_sizing_components = []

    for imported_component in imported_components:
        if "Sizing" in imported_component:
            imported_sizing_components.append(imported_component)

    defined_components = []
    for component_om_name in resources.DICTIONARY_CN:
        defined_components.append("Sizing" + resources.DICTIONARY_CN[component_om_name])

    assert set(imported_sizing_components) == set(defined_components)
