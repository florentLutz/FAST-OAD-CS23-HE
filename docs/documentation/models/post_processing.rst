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

Plots a financial parameter (e.g. the Net Present Value) against time in years, with an optional secondary y-axis
for another quantity (e.g. the number of aircraft delivered). Several designs can be plotted at once by passing lists
of file paths, or added one by one by passing an existing figure.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.financial_performance_curve import financial_performance_curve

    # Load the output file
    output_file = "path/to/output_file.yml"

    # Generate the financial performance curve
    financial_performance_curve(
        aircraft_file_path = output_file, # str or list of str
        name = "", # str or list of str
        plot_title = "title_of_the_plot",
        fig = None,
        file_formatter = None,
        period = 20,
        zero_reference_value = 0.0, # float, None
        discipline = "production", # "production", "operation"
        financial_parameter = "net_present_value", # str, list of str, None
        financial_parameter_axis_label = "NPV [$]",
        y_variable = None, # str, list of str, None
        y2_axis_label = "",
        fill_color = None, # str, list of str, None
        line_color = None, # str, list of str, None
        fill = True, # True, False
    )

| ``aircraft_file_path`` : Path to the output file, or list of paths to compare several designs.
| ``name`` : Legend name, or list of names matching ``aircraft_file_path``.
| ``plot_title`` : Title of the figure.
| ``fig`` : Existing figure to add the curves to, a new one is created if ``None``.
| ``file_formatter`` : Formatter of the output file, FAST-OAD default format if ``None``.
| ``period`` : Number of years on the x-axis (0 to ``period``), must match the length of the plotted variables.
| ``zero_reference_value`` : Y value of the dashed reference line on the main axis (e.g. break-even), ``None`` to hide it.
| ``discipline`` : ``production`` or ``operation``, one for all financial parameters or one per parameter.
| ``financial_parameter`` : Plots ``data:cost:<discipline>:<financial_parameter>``, ``None`` to plot ``y_variable`` only.
| ``financial_parameter_axis_label`` : Label of the main (left) y-axis.
| ``y_variable`` : Full name of an extra variable, e.g. ``data:cost:production:aircraft_delivered``.
| ``y2_axis_label`` : Label of the secondary (right) y-axis, also added to the secondary curve legend name.
| ``fill_color`` : Fill color, one for all designs or one per design, Plotly palette if ``None``.
| ``line_color`` : Line color, one for all designs or one per design, Plotly palette if ``None``.
| ``fill`` : Shades the area between the main curve and zero.

.. note::

    At most two y variables can be plotted: the financial parameter first (left axis), then ``y_variable`` (right
    axis). Variables containing ``delivery_count`` are filled with zeros for the development years.

*************************
LCC Cost Bar Chart Simple
*************************

Compares the cost items of several aircraft as grouped bars, one group per cost item. Only the positive items are
shown, so cost reductions (e.g. the landing gear cost reduction) are hidden. The power train component costs are
found automatically in the output file.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.lcc_cost import lcc_cost_bar_chart_simple

    lcc_cost_bar_chart_simple(
        aircraft_file_paths = ["path/to/output_file_1.yml", "path/to/output_file_2.yml"],
        names_aircraft = ["aircraft_1", "aircraft_2"],
        cost_type = "production", # "production", "operation"
        graph_title = None,
        item_filter_list = [],
        color_list = [],
    )

| ``aircraft_file_paths`` : List of paths to the output files.
| ``names_aircraft`` : List of aircraft names, used in the legend.
| ``cost_type`` : ``production`` (cost per unit, USD) or ``operation`` (annual cost, USD/yr).
| ``graph_title`` : Title of the figure, built from the aircraft names if ``None``.
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

    lcc_power_train_cost_breakdown(
        aircraft_file_paths = ["path/to/output_file_1.yml", "path/to/output_file_2.yml"], # str or list of str
        names_aircraft = ["aircraft_1", "aircraft_2"], # str or list of str
        cost_type = "production", # "production", "operation"
        group_by = "component", # "component", "type"
        exclude_list = [],
        graph_title = None,
        legend_rename = {},
        aggregate_and_sort_contributor = {},
        cutoff_criteria = None,
        show_total = True, # True, False
        reference_value = None,
        reference_label = "Reference",
        color_dict = {},
    )

| ``aircraft_file_paths`` : Path to the output file, or list of paths.
| ``names_aircraft`` : Aircraft name, or list of names, used as x-axis labels.
| ``cost_type`` : ``production`` (purchase cost, USD) or ``operation`` (annual operational cost, USD/yr).
| ``group_by`` : ``component`` for one segment per id (``motor_1``), ``type`` to sum ids of a same type (``PMSM``).
| ``exclude_list`` : Components to leave out, by type (``PEMFC_stack``) or id (``pemfc_stack_1``).
| ``graph_title`` : Title of the figure, default title if ``None``.
| ``legend_rename`` : ``{legend name: new name}`` to rename legend entries.
| ``aggregate_and_sort_contributor`` : ``{group name: name or list of names}`` to sum under one segment. Groups are
  stacked first, from the bottom, in the given order.
| ``cutoff_criteria`` : Items below this % of the total on every aircraft are merged into ``Others``.
| ``show_total`` : Writes the total above each bar.
| ``reference_value`` : Y value of a dashed reference line (e.g. a budget), ``None`` to hide it.
| ``reference_label`` : Label of the reference line.
| ``color_dict`` : ``{item: color}`` to force the color of some segments.

Component types and ids can be mixed in ``exclude_list`` and ``aggregate_and_sort_contributor``; an unknown name
raises a ``ValueError``.

***************************
LCC Cost Bar Chart Advanced
***************************

Plots all the cost items as one stacked bar per aircraft, with one segment per item. The power train total is
replaced by its components, the MSP is left out since it is not a share of the total, and cost reductions are drawn
below zero so that the totals are net values.

.. code:: python

    from fast_oad_cs23_he.src.fastga_he.gui.lcc_cost import lcc_cost_bar_chart_advanced

    lcc_cost_bar_chart_advanced(
        aircraft_file_paths = ["path/to/output_file_1.yml", "path/to/output_file_2.yml"], # str or list of str
        names_aircraft = ["aircraft_1", "aircraft_2"], # str or list of str
        cost_type = "production", # "production", "operation"
        power_train_detail = "component", # "component", "type", "aggregate"
        item_filter_list = [],
        exclude_list = [],
        normalisation_factors = [],
        graph_title = None,
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

| ``aircraft_file_paths`` : Path to the output file, or list of paths.
| ``names_aircraft`` : Aircraft name, or list of names, used as x-axis labels.
| ``cost_type`` : ``production`` (cost per unit, USD) or ``operation`` (annual cost, USD/yr).
| ``power_train_detail`` : ``component`` (one segment per id), ``type`` (one per type) or ``aggregate`` (a single
  ``Powertrain`` segment).
| ``item_filter_list`` : Items to keep. All by default.
| ``exclude_list`` : Items to leave out, ``Powertrain`` for the whole power train.
| ``normalisation_factors`` : One divisor per aircraft (e.g. number of seats for a cost per seat).
| ``graph_title`` : Title of the figure, default title if ``None``.
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