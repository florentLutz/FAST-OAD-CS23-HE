# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import plotly.graph_objects as go

from openmdao.utils.units import convert_units

from fastoad.io import VariableIO

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator, PT_DATA_PREFIX


def power_train_mass_breakdown(
    aircraft_file_path,
    power_train_file_path,
    file_formatter=None,
):
    """
    Returns a figure sunburst plot of the mass breakdown of the power train.

    :param aircraft_file_path: path of data file
    :param power_train_file_path: path of the power train file
    :param file_formatter: the formatter that defines the format of data file. If not provided,
                           default format will be assumed.
    :return: sunburst plot figure
    """

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)
    variables_names = configurator.get_mass_element_lists()

    component_level_breakdown = {}
    component_types = []

    # Create a dictionary with names and value and the right units, also identify the component
    # type so we can do a breakdown by type
    for variable_name in variables_names:
        component_type = variable_name.replace(PT_DATA_PREFIX, "").split(":")[0]
        component_types.append(component_type)

        variable_old_value = variables[variable_name].value[0]
        variable_old_unit = variables[variable_name].units
        variable_new_value = round(convert_units(variable_old_value, variable_old_unit, "kg"), 6)

        component_level_breakdown[variable_name] = (variable_new_value, component_type)

    power_train_weight = round(
        convert_units(
            variables[PT_DATA_PREFIX + "mass"].value[0],
            variables[PT_DATA_PREFIX + "mass"].units,
            "kg",
        ),
        6,
    )

    # Get unique components type
    component_types_unique = list(set(component_types))
    component_types_level_breakdown = dict(
        zip(component_types_unique, np.zeros(len(component_types_unique)))
    )

    power_train_weight_check = 0.0

    for component in component_level_breakdown:
        component_weight = component_level_breakdown[component][0]
        component_type = component_level_breakdown[component][1]
        component_types_level_breakdown[component_type] += component_weight
        power_train_weight_check += component_weight

    if power_train_weight != round(power_train_weight_check, 6):
        print(
            "Difference found between power train weight and the sum of the components, "
            "" + str(power_train_weight) + " vs " + str(round(power_train_weight_check, 6))
        )
        power_train_weight = round(power_train_weight_check, 6)

    # Create lists for the sunburst and start filling them with the power train mass value
    pt_weight_label = "Power train weight" + "<br> " + str(power_train_weight) + " [kg]"
    figure_labels = [pt_weight_label]
    figure_parents = [""]
    figure_values = [power_train_weight]

    for component_type in component_types_level_breakdown:
        component_type_weight = component_types_level_breakdown[component_type]
        component_type_label = component_type + "<br> " + str(component_type_weight) + " [kg]"
        figure_labels.append(component_type_label)
        figure_parents.append(pt_weight_label)
        figure_values.append(component_type_weight)

    for component in component_level_breakdown:
        component_weight = component_level_breakdown[component][0]
        component_type = component_level_breakdown[component][1]
        component_type_weight = component_types_level_breakdown[component_type]
        component_type_label = component_type + "<br> " + str(component_type_weight) + " [kg]"

        component_label = component.replace(PT_DATA_PREFIX + component_type + ":", "").split(":")[0]

        figure_labels.append(component_label)
        figure_parents.append(component_type_label)
        figure_values.append(component_weight)

    # Plot figure
    fig = go.Figure()
    fig.add_trace(
        go.Sunburst(
            labels=figure_labels,
            parents=figure_parents,
            values=figure_values,
            branchvalues="total",
        ),
    )
    fig.update_traces(maxdepth=2, selector=dict(type="sunburst"))
    fig.update_layout(title_text="Power train mass breakdown", title_x=0.5)

    return fig
