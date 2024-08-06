# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib

import pandas as pd

import plotly.graph_objects as go

THERMODYNAMIC_POWER_COLUMN_NAME = "Design Power (kW)"


def identify_design(df: pd.DataFrame):
    """
    We'll assume that the chance of two designs in the data having the same design power is
    minimal (though not theoretically impossible) so we'll identify designs by their power.
    """

    design_powers = df[THERMODYNAMIC_POWER_COLUMN_NAME].to_list()
    unique_design_powers = list(set(design_powers))

    return unique_design_powers


if __name__ == "__main__":
    plot = False

    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_folder_path = parent_folder / "data"

    result_file_path_max_power = data_folder_path / "max_power.csv"

    existing_data = pd.read_csv(result_file_path_max_power, index_col=0)

    designs = identify_design(existing_data)

    if plot:
        # Show the max power curves for the first design
        first_design_dataframe = existing_data.loc[
            existing_data[THERMODYNAMIC_POWER_COLUMN_NAME] == designs[0]
        ]

        altitude_list = list(set(first_design_dataframe["Altitude (ft)"].to_list()))
        mach_list = list(set(first_design_dataframe["Mach (-)"].to_list()))

        max_power_opr = first_design_dataframe["Max Power OPR Limit (kW)"].to_list()
        max_power_itt = first_design_dataframe["Max Power ITT Limit (kW)"].to_list()

        fig = go.Figure()

        altitude_list.sort()

        for idx, alt in enumerate(altitude_list):
            dataframe_current_alt = first_design_dataframe.loc[
                first_design_dataframe["Altitude (ft)"] == alt
            ]

            scatter_current_alt_opr_limit = go.Scatter(
                x=dataframe_current_alt["Mach (-)"].to_list(),
                y=dataframe_current_alt["Max Power OPR Limit (kW)"].to_list(),
                mode="lines+markers",
                name="Max power OPR limit",
                legendgroup=str(alt),
                legendgrouptitle_text=str(alt),
            )
            fig.add_trace(scatter_current_alt_opr_limit)
            scatter_current_alt_itt_limit = go.Scatter(
                x=dataframe_current_alt["Mach (-)"].to_list(),
                y=dataframe_current_alt["Max Power ITT Limit (kW)"].to_list(),
                mode="lines+markers",
                name="Max power ITT limit",
                legendgroup=str(alt),
            )
            fig.add_trace(scatter_current_alt_itt_limit)

        fig.show()
