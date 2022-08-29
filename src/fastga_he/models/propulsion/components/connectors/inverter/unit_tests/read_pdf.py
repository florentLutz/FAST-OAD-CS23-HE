# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from os import path, listdir
import pandas as pd
import numpy as np
import tabula


def parsing_number(number_to_parse: str):

    number_of_number = how_many_number(number_to_parse)
    if number_of_number == 1:
        return [float(number_to_parse)]
    elif number_of_number == 2:
        split_number = number_to_parse.split(".")
        second_number = float(split_number[-2][-1] + "." + split_number[-1])
        if len(split_number) == 2:
            first_number = float(split_number[-2][0:-1])
        else:
            first_number = float(split_number[-3] + "." + split_number[-2][0:-1])
        return [first_number, second_number]
    else:
        split_number = number_to_parse.split(".")
        third_number = float(split_number[-2][-1] + "." + split_number[-1])
        second_number = float(split_number[-3][-1] + "." + split_number[-2][0:-1])
        if len(split_number) == 3:
            first_number = float(split_number[-3][0:-1])
        else:
            first_number = float(split_number[-4][-1] + "." + split_number[-3][0:-1])
        return [first_number, second_number, third_number]


def how_many_number(number_to_parse: str):

    split_number = number_to_parse.split(".")
    if len(split_number) == 1 or (len(split_number) == 2 and len(split_number[0]) == 1):
        return 1
    elif len(split_number) == 2 or (len(split_number) == 3 and len(split_number[0]) == 1):
        return 2
    else:
        return 3


def add_module_to_database(file_path, results_df, columns_title):
    tables = tabula.read_pdf(file_path, pages=[1, 2])
    module_name = tables[0].columns[0]

    for table in tables:
        if "Absolute Maximum Ratings" in table.columns:
            column = table["Absolute Maximum Ratings"]
            idx_ic_igbt = np.where(column == "ICnom")[0][0]
            idx_max_voltage = np.where(column == "VCES")[0][0]
            ic_igbt = table.loc[idx_ic_igbt, "Unnamed: 1"]
            max_voltage = table.loc[idx_max_voltage, "Unnamed: 1"]
        elif "Characteristics" in table.columns:
            column = table["Characteristics"]
            if "IGBT" in column.to_numpy():
                idx_voltage_drop_igbt = np.where(column == "VCE0")[0][0]
                un_parsed_voltage_drop_igbt = table.loc[idx_voltage_drop_igbt, "Unnamed: 2"]
                voltage_drop_igbt = parsing_number(un_parsed_voltage_drop_igbt)[-1]
                idx_resistance_igbt = np.where(column == "rCE")[0][0]
                un_parsed_resistance_igbt = table.loc[idx_resistance_igbt, "Unnamed: 2"]
                resistance_igbt = parsing_number(un_parsed_resistance_igbt)[-1]
                idx_thermal_resistance_igbt = np.where(column == "Rth(j-c)")[0][0]
                un_parsed_thermal_resistance_igbt = table.loc[
                    idx_thermal_resistance_igbt, "Unnamed: 1"
                ]
                thermal_resistance_igbt = parsing_number(un_parsed_thermal_resistance_igbt)[-1]
            elif "Inverse diode" in column.to_numpy():
                idx_voltage_drop_diode = np.where(column == "VF0")[0][0]
                un_parsed_voltage_drop_diode = table.loc[idx_voltage_drop_diode, "Unnamed: 2"]
                voltage_drop_diode = parsing_number(un_parsed_voltage_drop_diode)[-1]
                idx_resistance_diode = np.where(column == "rF")[0][0]
                un_parsed_resistance_diode = table.loc[idx_resistance_diode, "Unnamed: 2"]
                resistance_diode = parsing_number(un_parsed_resistance_diode)[-1]
                idx_thermal_resistance_diode = np.where(column == "Rth(j-c)")[0][0]
                un_parsed_thermal_resistance_diode = table.loc[
                    idx_thermal_resistance_diode, "Unnamed: 1"
                ]
                thermal_resistance_diode = parsing_number(un_parsed_thermal_resistance_diode)[-1]

                # Module will also be there hopefully
                idx_module_weight = np.where(column == "w")[0][0]
                un_parsed_module_weight = table.loc[idx_module_weight, "Unnamed: 1"]
                module_weight = parsing_number(un_parsed_module_weight)[-1]
                idx_module_thermal_resistance = np.where(column == "Rth(c-s)")[0][0]
                un_parsed_module_thermal_resistance = table.loc[
                    idx_module_thermal_resistance, "Unnamed: 1"
                ]
                module_thermal_resistance = parsing_number(un_parsed_module_thermal_resistance)[-1]

    local_results_df = pd.DataFrame(
        [
            [
                module_name,
                ic_igbt,
                max_voltage,
                voltage_drop_igbt,
                resistance_igbt,
                thermal_resistance_igbt,
                voltage_drop_diode,
                resistance_diode,
                thermal_resistance_diode,
                module_weight,
                module_thermal_resistance,
            ]
        ],
        columns=columns_title,
    )

    results_df = pd.concat([results_df, local_results_df])

    return results_df


columns_title_final = [
    "Module name",
    "Current caliber (IGBT)",
    "Maximum voltage (IGBT)",
    "Voltage drop (IGBT)",
    "Dynamic resistance (IGBT)",
    "Thermal resistance (IGBT)",
    "Voltage drop (diode)",
    "Dynamic resistance (diode)",
    "Thermal resistance (diode)",
    "Module weight",
    "Module-level resistance",
]

data_folder = "D:/fl.lutz/Documents/Biblio/Electric Aircraft Design/Inverter/IGBT_7"

final_results_df = pd.DataFrame(columns=columns_title_final)

filenames = listdir(data_folder)
for file_name in filenames:
    data_file = path.join(data_folder, file_name)
    try:
        final_results_df = add_module_to_database(data_file, final_results_df, columns_title_final)
    except:
        print("Could not read " + data_file)

final_results_df.to_csv(
    "D:/fl.lutz/Documents/Biblio/Electric Aircraft Design/Inverter/IGBT_7/database_igbt7.csv"
)
