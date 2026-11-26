.. _post_processing:

================================
Post-processing Helper Functions
================================

FAST-OAD-CS23-HE provides a set of post-processing helper functions that can be used to extract and analyze the results
from the output file and the powertrain watcher file, which allows user to visualize and interact with the results
in a more intuitive way.

.. contents::


***************************
Financial Performance Curve
***************************

Plots a financial model variable (e.g. the Net Present Value) against time in years, with an optional secondary
y-axis for another quantity (e.g. the annual delivery count). Several designs can be plotted at once by passing a list
of file paths, or added one by one by passing the returned figure back in with ``fig``.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.financial_performance_curve import financial_performance_curve

    # Load the output file
    output_file = "path/to/output_file.xml"

    # Generate the financial performance curve
    fig = financial_performance_curve(
        aircraft_file_path = output_file, # str or list of str
        name = "", # str or list of str
        fig = None,
        file_formatter = None,
        period = 20,
        zero_reference_value = 0.0, # float, None
        discipline = "production", # "production", "operation", or a list of them
        y_variable = "net_present_value", # str or list of up to two str
        y_axis_label = "NPV [$]", # str or list of str
        line_color = None, # str, list of str, None
        fill = True, # True, False
    )

    fig.show()

| ``aircraft_file_path`` : Path to the output file, or list of paths to compare several designs.
| ``name`` : Legend name, one for all designs or one per design.
| ``fig`` : Existing figure to add the curves to, a new one is created if ``None``.
| ``file_formatter`` : Formatter of the output file, FAST-OAD default format if ``None``.
| ``period`` : Number of years on the x-axis (0 to ``period``). The plotted variables must have ``period + 1`` values.
| ``zero_reference_value`` : Y value of the dashed reference line on the main axis (e.g. break-even), ``None`` to hide it.
| ``discipline`` : ``production`` or ``operation``, one for all y variables or a list with one per y variable. Only used for short names in ``y_variable``.
| ``y_variable`` : One or two variables to plot. A short name without ``:`` is considered as financial model variables, read as ``data:cost:<discipline>:<name>``. The full variable name is read as it is.
| ``y_axis_label`` : Axis label(s), in the same order as ``y_variable``. A single string labels the main axis only; a list must give one label per plotted variable.
| ``line_color`` : RGBA line and marker color. Designs without a color (``None``or past the end of a shorter list) get the next color of the Plotly palette.
| ``fill`` : Shades the area between the main curve and zero, in the Plotly palette color at 0.3 opacity.

.. note::

    At most two y variables can be plotted: the first one on the main (left) axis, the second one on the secondary
    (right) axis, as a dashed curve with diamond markers in the same color as the main curve.

List of available financial model variables
===========================================

.. raw:: html

   <div style="display: flex; justify-content: center;">

================================================  ================================================
Production                                          Operation
================================================  ================================================
annual_cash_flow                                    airfare_compound_annual_gain_factor
net_present_value                                   annual_cash_flow
profitability_index                                 net_present_value
annual_delivery_count                               profitability_index
cumulative_annual_delivery_count                    annual_revenue_projection
annual_gross_profit                                 annual_energy_cost_projection
maturity_discount                                   annual_hydrocarbon_fuel_cost_projection
project_development_cost_distribution               annual_hydrogen_fuel_cost_projection
project_development_cost_cumulative_distribution    annual_electricity_cost_projection
\                                                   hydrocarbon_fuel_compound_annual_gain_factor
\                                                   hydrogen_fuel_compound_annual_gain_factor
\                                                   electricity_compound_annual_gain_factor
================================================  ================================================

.. raw:: html

   </div>

*************************
LCC Cost Bar Chart Simple
*************************

Compares the cost items of several aircraft as grouped bars, one group per cost item. Only the positive items are
shown, so cost reductions (e.g. the landing gear cost reduction) are hidden. The powertrain component costs are
found automatically in the output file.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.lcc_cost import lcc_cost_bar_chart_simple

    fig = lcc_cost_bar_chart_simple(
        aircraft_file_paths = ["path/to/output_file_1.xml", "path/to/output_file_2.xml"],
        names_aircraft = ["aircraft_1", "aircraft_2"],
        cost_type = "production", # "production", "operation"
        item_filter_list = [],
        color_list = ["blue", "red"],
    )

    fig.show()

| ``aircraft_file_paths`` : List of paths to the output files.
| ``names_aircraft`` : List of aircraft names, used in the legend.
| ``cost_type`` : ``production`` (cost per unit, USD) or ``operation`` (annual cost, USD/yr).
| ``item_filter_list`` : Cost items to show, by label (``Fuel``) or variable name (``annual_fuel_cost``). All by default.
| ``color_list`` : One bar color per aircraft, Plotly colors by default.

*****************************
LCC Powertrain Cost Breakdown
*****************************

Plots the power train cost as one stacked bar per aircraft, with one segment per component: the purchase cost for
``production`` and the annual operational cost for ``operation``. Components are found by scanning the
``data:propulsion:he_power_train:<type>:<id>:purchase_cost`` and ``:operational_cost`` variables of the output file.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.lcc_cost import lcc_power_train_cost_breakdown

    fig = lcc_power_train_cost_breakdown(
        aircraft_file_paths = ["path/to/output_file_1.xml", "path/to/output_file_2.xml"], # str or list of str
        names_aircraft = ["aircraft_1", "aircraft_2"], # str or list of str
        cost_type = "production", # "production", "operation"
        group_by = "component", # "component", "type"
        exclude_list = [],
        legend_rename = {},
        aggregate_and_sort_contributor = {},
        cutoff_criteria = None,
        show_total = True, # True, False
        reference_value = None,
        reference_label = "Reference",
        color_dict = {},
    )

    fig.show()

| ``aircraft_file_paths`` : Path to the output file, or list of paths.
| ``names_aircraft`` : Aircraft name, or list of names, used as x-axis labels.
| ``cost_type`` : ``production`` (purchase cost, USD) or ``operation`` (annual operational cost, USD/yr).
| ``group_by`` : ``component`` for one segment per id (``motor_1``), ``type`` to sum ids of a same type (``PMSM``).
| ``exclude_list`` : Components to leave out, by type (``PEMFC_stack``) or id (``pemfc_stack_1``).
| ``legend_rename`` : ``{legend name: new name}`` to rename legend entries.
| ``aggregate_and_sort_contributor`` : ``{group name: name or list of names}`` to sum under one segment. Groups are stacked first, from the bottom, in the given order.
| ``cutoff_criteria`` : Items below this % of the total on every aircraft are merged into ``Others``.
| ``show_total`` : Writes the total above each bar.
| ``reference_value`` : Y value of a dashed reference line (e.g. a budget), ``None`` to hide it.
| ``reference_label`` : Label of the reference line.
| ``color_dict`` : ``{item: color}`` to force the color of some segments.

Component types and ids can be mixed in ``exclude_list`` and ``aggregate_and_sort_contributor``; an unknown name raises
a ``ValueError``.

***************************
LCC Cost Bar Chart Advanced
***************************

Plots all the cost items as one stacked bar per aircraft, with one segment per item. The power train total is
replaced by its components, the MSP is left out since it is not a share of the total, and cost reductions are drawn
below zero so that the totals are net values.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.lcc_cost import lcc_cost_bar_chart_advanced

    fig = lcc_cost_bar_chart_advanced(
        aircraft_file_paths = ["path/to/output_file_1.xml", "path/to/output_file_2.xml"], # str or list of str
        names_aircraft = ["aircraft_1", "aircraft_2"], # str or list of str
        cost_type = "production", # "production", "operation"
        power_train_detail = "component", # "component", "type", "aggregate"
        item_filter_list = [],
        exclude_list = [],
        normalisation_factors = [],
        y_axis_title = None,
        value_unit = None,
        legend_rename = {},
        aggregate_and_sort_contributor = {},
        cutoff_criteria = None,
        sort_by_size = True, # True, False
        show_total = True, # True, False
        reference_value = None,
        reference_label = "Reference",
        color_dict = {},
    )

    fig.show()

| ``aircraft_file_paths`` : Path to the output file, or list of paths.
| ``names_aircraft`` : Aircraft name, or list of names, used as x-axis labels.
| ``cost_type`` : ``production`` (cost per unit, USD) or ``operation`` (annual cost, USD/yr).
| ``power_train_detail`` : ``component`` (one segment per id), ``type`` (one per type) or ``aggregate`` (a
  single ``Powertrain`` segment).
| ``item_filter_list`` : Items to keep. All by default.
| ``exclude_list`` : Items to leave out, ``Powertrain`` for the whole power train.
| ``normalisation_factors`` : One divisor per aircraft (e.g. number of seats for a cost per seat).
| ``y_axis_title`` : Title of the y-axis, default title if ``None``.
| ``value_unit`` : Unit shown in the totals, hover text and reference line, ``USD`` or ``USD/yr`` if ``None``.
| ``legend_rename`` : ``{legend name: new name}`` to rename legend entries.
| ``aggregate_and_sort_contributor`` : ``{group name: name or list of names}`` to sum under one segment. Groups are
  stacked first, from the bottom, in the given order.
| ``cutoff_criteria`` : Items below this % of the total on every aircraft are merged into ``Others``.
| ``sort_by_size`` : Stacks the biggest items at the bottom, otherwise keeps the default item order.
| ``show_total`` : Writes the net total above each bar.
| ``reference_value`` : Y value of a dashed reference line (e.g. a budget or the revenue), ``None`` to hide it.
| ``reference_label`` : Label of the reference line.
| ``color_dict`` : ``{item: color}`` to force the color of some segments.

.. note::

    Items can be named by label (``Tooling``), variable name (``tooling_cost_per_unit``), component type or id
    (``motor_1``) or ``Powertrain``. An unknown name in ``exclude_list`` or ``aggregate_and_sort_contributor``
    raises a ``ValueError``. A warning is logged if the power train components don't add up to the power train
    total in the file.