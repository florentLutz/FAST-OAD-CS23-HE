"""API."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Union

import numpy as np
import openmdao.api as om

import fastoad.api as oad


def list_inputs_metadata(component: Union[om.ExplicitComponent, om.Group]) -> tuple:
    """
    Reads all variables from a component/problem and returns inputs name and metadata as a
    list.
    """

    prob = oad.FASTOADProblem()
    model = prob.model
    model.add_subsystem("component", component, promotes=["*"])

    prob_copy = deepcopy(prob)

    var_copy_shape_name_list = []
    var_copy_shape_list = []

    try:
        prob_copy.setup()
    except RuntimeError:
        # noinspection PyProtectedMember
        vars_metadata = oad.FASTOADProblem()._get_undetermined_dynamic_vars_metadata(prob_copy)
        if vars_metadata:
            # If vars_metadata is empty, it means the RuntimeError was not because
            # of dynamic shapes, and the incoming self.setup() will raise it.
            ivc = om.IndepVarComp()
            for name, meta in vars_metadata.items():
                # We use a (2,)-shaped array as value here. This way, it will be easier to
                # identify dynamic-shaped data in an input file generated from current problem.
                var_copy_shape_name_list.append(name)
                var_copy_shape_list.append(meta["copy_shape"])
                ivc.add_output(name, [np.nan, np.nan], units=meta["units"])
            prob.model.add_subsystem("temp_shaper", ivc, promotes=["*"])

    variables = prob_copy.model.get_io_metadata(iotypes="input")
    variables_out = prob_copy.model.get_io_metadata(iotypes="output")

    # We first identify which variables are output so we can not add them to the list of input
    # because for some weird reasons it show the inputs of every components not just the group
    # inputs
    var_outputs = []
    for variable_out_name in variables_out:

        variable = variables_out[variable_out_name]
        var_prom_name = variable["prom_name"]

        # We filter out the auto_ivc and the variables from the shape
        is_ivc = "_auto_ivc" in variable_out_name
        is_shaper = "fastoad_shaper" in variable_out_name

        if var_prom_name not in var_outputs and not is_ivc and not is_shaper:
            var_outputs.append(var_prom_name)

    var_inputs = []
    var_units = []
    var_value = []
    var_shape = []
    var_shape_by_conn = []
    var_copy_shape = []

    for variable_name in variables:

        variable = variables[variable_name]
        var_prom_name = variable["prom_name"]

        # We check that it has not been added already or that it is not an output of another module
        if var_prom_name not in var_inputs and var_prom_name not in var_outputs:
            var_inputs.append(variable["prom_name"])
            var_units.append(variable["units"])
            var_value.append(prob_copy.get_val(variable_name, variable["units"]))
            var_shape.append(variable["shape"])
            var_shape_by_conn.append(variable["shape_by_conn"])
            var_copy_shape.append(variable["copy_shape"])

    return var_inputs, var_units, var_value, var_shape, var_shape_by_conn, var_copy_shape
