"""
Convenience functions for helping tests
"""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging
import os.path as pth
from copy import deepcopy
from typing import Union, List

import numpy as np

import openmdao.api as om
from openmdao.core.system import System


# noinspection PyProtectedMember
from fastoad.module_management.service_registry import _RegisterOpenMDAOService
import fastoad.api as oad

from fastoad.io import VariableIO
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

_LOGGER = logging.getLogger(__name__)


def run_system(
    component: System,
    input_vars: om.IndepVarComp,
    setup_mode="auto",
    add_solvers=False,
    check=False,
):
    """Runs and returns an OpenMDAO problem with provided component and data"""
    problem = oad.FASTOADProblem(reports=False)
    model = problem.model
    model.add_subsystem("inputs", input_vars, promotes=["*"])
    model.add_subsystem("component", component, promotes=["*"])
    if add_solvers:
        # noinspection PyTypeChecker
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.DirectSolver()

    if check:
        print("\n")

    problem.setup(mode=setup_mode, check=check)
    variables = [
        var.name
        for var in oad.VariableList.from_problem(problem, io_status="inputs")
        if np.any(np.isnan(var.val))
    ]
    assert not variables, "These inputs are not provided: %s" % variables

    problem.run_model()

    return problem


# FIXME: problem to be solved on the register
def register_wrappers():
    """Register all the wrappers from models"""
    path, folder_name = pth.dirname(__file__), None
    unsplit_path = path
    while folder_name != "models":
        unsplit_path = path
        path, folder_name = pth.split(path)
    _RegisterOpenMDAOService.explore_folder(unsplit_path)


def get_indep_var_comp(var_names: List[str], test_file: str, xml_file_name: str) -> om.IndepVarComp:
    """Reads required input data from xml file and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(test_file), "data", xml_file_name))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()

    return ivc


class VariableListLocal(oad.VariableList):
    @classmethod
    def from_system(cls, system: System) -> "oad.VariableList":
        """
        Creates a VariableList instance containing inputs and outputs of a an OpenMDAO System.
        The inputs (is_input=True) correspond to the variables of IndepVarComp
        components and all the unconnected variables.

        Warning: setup() must NOT have been called.

        In the case of a group, if variables are promoted, the promoted name
        will be used. Otherwise, the absolute name will be used.

        :param system: OpenMDAO Component instance to inspect
        :return: VariableList instance.
        """

        problem = oad.FASTOADProblem(reports=False)
        if isinstance(system, om.Group):
            problem.model = deepcopy(system)
        else:
            # problem.model has to be a group
            problem.model.add_subsystem("comp", deepcopy(system), promotes=["*"])
        problem.setup()
        return VariableListLocal.from_problem(problem, use_initial_values=True)


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """Reads input variables from a component/problem and return as a list"""
    # register_wrappers()
    if isinstance(component, om.Group):
        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
    variables = VariableListLocal.from_system(component)
    input_names = [var.name for var in variables if var.is_input]

    return input_names
