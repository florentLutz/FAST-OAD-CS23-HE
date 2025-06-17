#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib

from typing import List

import pytest

import openmdao.api as om
from fastoad.io import VariableIO

from tests.testing_utilities import run_system, list_inputs
from ..lca import LCA

XML_FILE = "data.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data_lca_pipistrel"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results_lca_pipistrel"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def local_get_indep_var_comp(var_names: List[str], xml_file_name: str) -> om.IndepVarComp:
    """Reads required input data from xml file and returns an IndepVarcomp() instance"""
    reader = VariableIO(DATA_FOLDER_PATH / xml_file_name)
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()

    return ivc


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_ef_fr_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_european_mix():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly_eu_mix.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_ef_eu_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_club():
    input_file_name = "pipistrel_club_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_club_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The fuel stored in the tanks does not consider the fuel necessary for takeoff. This is a
    # shortcoming of the code for now

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_club_lca_out_ef_fr_mix.xml"
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        0.005710905517643169, rel=1e-3
    )
    assert problem.get_val(
        "data:LCA:operation:he_power_train:gasoline:mass_per_fu", units="kg"
    ) == pytest.approx(11.59, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:high_rpm_ICE:ice_1:CO2_per_fu", units="kg"
    ) == pytest.approx(23.19468787840188, rel=1e-3)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_club_recipe():
    input_file_name = "pipistrel_club_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_club_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_club_lca_out_recipe_fr_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_recipe():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_fr_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_club_iw():
    """
    Run this at you own risks and perils. It runs but it takes more than 45 min, whereas other
    LCIA method only take 3 min or so
    """
    input_file_name = "pipistrel_club_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_club_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        weighting=True,
        ecoinvent_version="3.10.1",
        impact_assessment_method="IMPACT World+ v2.0.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_club_lca_out_iw_fr_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_iw():
    """
    Run this at you own risks and perils. It runs but it takes more than 45 min, whereas other
    LCIA method only take 3 min or so
    """
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.10.1",
        impact_assessment_method="IMPACT World+ v2.0.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_iw_fr_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_heavy_recipe():
    input_file_name = "pipistrel_heavy_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_fr_mix.xml"
    )
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_european_mix_btf():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly_eu_mix.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )
    inputs_list = list_inputs(component)
    inputs_list.remove("data:environmental_impact:buy_to_fly:composite")

    ivc = local_get_indep_var_comp(inputs_list, input_file_name)
    ivc.add_output("data:environmental_impact:buy_to_fly:composite", val=1.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_eu_mix_btf.xml"
    )
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_heavy_european_mix_btf():
    input_file_name = "pipistrel_heavy_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly_eu_mix.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="train",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )
    inputs_list = list_inputs(component)
    inputs_list.remove("data:environmental_impact:buy_to_fly:metallic")

    ivc = local_get_indep_var_comp(inputs_list, input_file_name)
    ivc.add_output("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_eu_mix_btf.xml"
    )
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_heavy_french_mix_btf():
    input_file_name = "pipistrel_heavy_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly_eu_mix.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )
    inputs_list = list_inputs(component)
    inputs_list.remove("data:environmental_impact:buy_to_fly:metallic")

    ivc = local_get_indep_var_comp(inputs_list, input_file_name)
    ivc.add_output("data:environmental_impact:buy_to_fly:metallic", val=7.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_fr_mix_btf.xml"
    )
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_recipe_with_btf():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )

    inputs_list = list_inputs(component)
    inputs_list.remove("data:environmental_impact:buy_to_fly:composite")

    ivc = local_get_indep_var_comp(inputs_list, input_file_name)
    ivc.add_output("data:environmental_impact:buy_to_fly:composite", val=1.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.run_model()
    problem.output_file_path = (
        RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_recipe_fr_mix_btf.xml"
    )
    problem.write_outputs()
