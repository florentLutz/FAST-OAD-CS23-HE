#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import os
import os.path as pth
import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import fastoad.api as oad
import pandas as pd

import plotly.graph_objects as go

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data_time_sensitivity")
RESULTS_TIME_SENSITIVITY = pth.join(pth.dirname(__file__), "results_time_sensitivity")


def merge_lca_data_xml(design_problem_output_file_path, lca_data_file_path):
    design_problem_datafile = oad.DataFile(design_problem_output_file_path)
    lca_problem_datafile = oad.DataFile(lca_data_file_path)

    for variable in lca_problem_datafile:
        design_problem_datafile.append(variable)

    design_problem_datafile.save()


def tech_to_year(bed, lifespan):
    year_rounded = 2025 + 5 * (
        1.823088191771652e-05
        * ((395 - bed) ** 2 + 0.0051968503937007875 * (lifespan - 150) ** 2)
        // 5
    )

    return int(year_rounded)


def _find_neighbour(current_lifespan, current_bed):
    """
    Simplified function to look for neighbours. Given out case, it will be either the point on top
    (improved bed) or the point to the right (improved lifespan)
    """

    if current_bed == 395:
        improved_bed = 500
    else:
        improved_bed = current_bed + 100

    if current_lifespan == 150.0:
        improved_lifespan = 1000.0
    else:
        improved_lifespan = current_lifespan + 1000.0

    if current_bed == 1300:  # Best BED we explored so only neighbour is improved lifespan
        return [(improved_lifespan, current_bed)]

    if current_lifespan == 12000.0:
        return [(current_lifespan, improved_bed)]

    return [(improved_lifespan, current_bed), (current_lifespan, improved_bed)]


def test_design_electric_commuter_time_sensitivity():
    """
    This test explores the sensitivity of the single score of the electric K100 with Na-ion cell
    to the cell density and expected lifespan. It will sweep values fom current tech (395 Wh/kg,
    150 cycles) to futuristic techs (values representative of Li-S and SIB). Each time,
    the design range will be adjusted to what is possible given the cell energy density. This
    will be done based on the results obtained with the electric aircraft and their different
    cell chemistry (likely via a polynomial function). Meaning some difference in MTOW are
    expected. Payload will also be adjusted since on the original design, it was required to
    reduce the number of passenger. From 395 to 700 Wh/kg a linear variation with a floor will be
    taken. One thing to keep in mind is that the cell chemistry is a discrete variable,
    which I can't change as continuous variable like I plan to do on cell longevity and energy
    density. So the same manufacturing process and cell voltage will be used for all points of
    the upcoming graph. Therefore, what I will mark a Li-S and Si-NMC will not actually be those
    chemistry (so not the same results) but something close. As a future work, it would be
    interesting to look into the space created to see where each battery is actually located and
    change the processes accordingly. Technological parameter will be assumed to vary linearly from
    their 2025 state with the BED of 395 Wh/kg to their 2040 values once a BED of 700 Wh/kg is
    reached. This means the technological parameters other than BED and lifespan don't really scale
    past a certain point.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    design_problem_input_file_name = "design_problem_inputs.xml"
    design_problem_process_file_name = "design_problem.yml"
    # lca_problem_input_file_name = "lca_problem_inputs.xml"
    # lca_problem_process_file_name = "lca_problem.yml"

    # investigated_energy_densities = np.array([395.0, 700.0, 1300.0])  # As a proof of concept
    investigated_energy_densities = np.array(
        [395.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0]
    )
    # investigated_lifespans = np.array(
    #     [
    #         150.0,
    #         1000.0,
    #         2000.0,
    #         3000.0,
    #         4000.0,
    #         5000.0,
    #         6000.0,
    #         7000.0,
    #         8000.0,
    #         9000.0,
    #         10000.0,
    #         11000.0,
    #         12000.0,
    #     ]
    # )
    # investigated_lifespans = np.array(
    #     [
    #         150.0,
    #     ]
    # )

    for investigated_energy_density in investigated_energy_densities:
        design_range = (
            3.74362226e-04 * investigated_energy_density**2
            - 8.20577846e-02 * investigated_energy_density
            + 7.40029587e01
        )
        number_of_pax = np.floor(
            np.interp(investigated_energy_density, [395, 700, 1301], [6, 8, 8])
        )
        # C-rate needs to also be adjusted to avoid the cells being sized for power instead of
        # energy. We'll make that assumption and further discuss upon it in the defence.
        c_rate_caliber = np.interp(investigated_energy_density, [395, 700, 1301], [4.0, 1.2, 1])
        # Rest of the technological parameters will also be adjusted, not based on year (it would
        # take too long) but based on energy density, which is questionable.
        # Interestingly, initial results have shown that 2040 tech for power converter led to lower
        # efficiencies which cause a significant mass divergence on the battery packs
        switching_frequency = np.interp(
            investigated_energy_density, [395, 700, 1301], [12e3, 100e3, 100e3]
        )
        resistance_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.85, 0.85]
        )
        sw_losses_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.25, 0.25]
        )
        rr_losses_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.16667, 0.16667]
        )
        gate_voltage_igbt = np.interp(
            investigated_energy_density, [395, 700, 1301], [0.87, 2.61, 2.61]
        )
        gate_voltage_module = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.3, 4.2, 4.2]
        )
        tangential_stress = np.interp(
            investigated_energy_density, [395, 700, 1301], [50000, 70000, 70000]
        )
        current_density = np.interp(investigated_energy_density, [395, 700, 1301], [8.1, 20, 20])
        surface_current_density = np.interp(
            investigated_energy_density, [395, 700, 1301], [111.1, 155.5, 155.5]
        )
        nb_pole_pair = 2 * np.floor(
            np.interp(investigated_energy_density, [395, 700, 1301], [1, 2, 2])
        )

        # Since we are changing the way the coltage is computed, we need to slightly adjust the
        # energy density at low BED to match values used with the precise model.
        k_bed = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0635266808778578, 1.0, 1.0]
        )

        # The cell has a nominal voltage of 2.7 V
        cell_capacity_ref = investigated_energy_density * 11.7e-3 / 2.7

        configurator_design_problem = oad.FASTOADProblemConfigurator(
            pth.join(DATA_FOLDER_PATH, design_problem_process_file_name)
        )
        design_problem = configurator_design_problem.get_problem()

        # Load inputs
        design_problem_inputs = pth.join(DATA_FOLDER_PATH, design_problem_input_file_name)

        design_problem.write_needed_inputs(design_problem_inputs)
        design_problem.read_inputs()

        design_problem.model_options["*"] = {
            "cell_capacity_ref": cell_capacity_ref * k_bed,
            "cell_weight_ref": 11.7e-3,
            "reference_curve_current": [100.0, 1000.0, 3000.0, 3720.0],
            "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
        }

        design_problem.setup()

        design_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
            val=c_rate_caliber,
            units="h**-1",
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
            val=c_rate_caliber,
            units="h**-1",
        )
        design_problem.set_val(
            "data:TLAR:range",
            val=design_range,
            units="nmi",
        )
        design_problem.set_val(
            "data:TLAR:NPAX_design",
            val=number_of_pax,
        )
        design_problem.set_val(
            "data:geometry:cabin:seats:passenger:NPAX_max",
            val=number_of_pax,
        )

        # Switching frequencies
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )

        # Resistance scaling
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:igbt:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:igbt:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:igbt:k_resistance",
            val=resistance_scaling,
        )

        # Losses
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:k_igbt_switching_losses",
            val=sw_losses_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:k_diode_switching_losses",
            val=rr_losses_scaling,
        )

        # Gate voltages
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )

        # E-motor parameter
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress",
            units="N/m**2",
            val=tangential_stress,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:design_phase_current_density",
            units="A/m**2",
            val=current_density * 1e6,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:design_surface_current_density",
            units="kA/m",
            val=surface_current_density,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=nb_pole_pair
        )

        # Run the design problem
        design_problem.run_model()
        design_problem_output_file_name = str(int(investigated_energy_density)) + "_wh_per_kg.xml"
        design_problem.output_file_path = pth.join(
            pth.join(RESULTS_TIME_SENSITIVITY, "design"), design_problem_output_file_name
        )
        design_problem.write_outputs()


def test_design_electric_commuter_time_feasibility():
    """
    This test does mostly the same thing as the test above expect it just tests whether the
    design is merely feasible.
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    design_problem_input_file_name = "design_problem_inputs.xml"
    design_problem_process_file_name = "design_problem.yml"

    investigated_energy_densities = np.array([600.0])

    for investigated_energy_density in investigated_energy_densities:
        design_range = (
            3.74362226e-04 * investigated_energy_density**2
            - 8.20577846e-02 * investigated_energy_density
            + 7.40029587e01
        )
        number_of_pax = np.floor(
            np.interp(investigated_energy_density, [395, 700, 1301], [6, 8, 8])
        )
        # C-rate needs to also be adjusted to avoid the cells being sized for power instead of
        # energy. We'll make that assumption and further discuss upon it in the defence.
        c_rate_caliber = np.interp(investigated_energy_density, [395, 700, 1301], [4.0, 1.2, 1])
        # Rest of the technological parameters will also be adjusted, not based on year (it would
        # take too long) but based on energy density, which is questionable.
        # Interestingly, initial results have shown that 2040 tech for power converter led to lower
        # efficiencies which cause a significant mass divergence on the battery packs
        switching_frequency = np.interp(
            investigated_energy_density, [395, 700, 1301], [12e3, 100e3, 100e3]
        )
        resistance_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.85, 0.85]
        )
        sw_losses_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.25, 0.25]
        )
        rr_losses_scaling = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0, 0.16667, 0.16667]
        )
        gate_voltage_igbt = np.interp(
            investigated_energy_density, [395, 700, 1301], [0.87, 2.61, 2.61]
        )
        gate_voltage_module = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.3, 4.2, 4.2]
        )
        tangential_stress = np.interp(
            investigated_energy_density, [395, 700, 1301], [50000, 70000, 70000]
        )
        current_density = np.interp(investigated_energy_density, [395, 700, 1301], [8.1, 20, 20])
        surface_current_density = np.interp(
            investigated_energy_density, [395, 700, 1301], [111.1, 155.5, 155.5]
        )
        nb_pole_pair = 2 * np.floor(
            np.interp(investigated_energy_density, [395, 700, 1301], [1, 2, 2])
        )

        # Since we are changing the way the coltage is computed, we need to slightly adjust the
        # energy density at low BED to match values used with the precise model.
        k_bed = np.interp(
            investigated_energy_density, [395, 700, 1301], [1.0635266808778578, 1.0, 1.0]
        )

        # The cell has a nominal voltage of 2.7 V
        cell_capacity_ref = investigated_energy_density * 11.7e-3 / 2.7

        configurator_design_problem = oad.FASTOADProblemConfigurator(
            pth.join(DATA_FOLDER_PATH, design_problem_process_file_name)
        )
        design_problem = configurator_design_problem.get_problem()

        # Load inputs
        design_problem_inputs = pth.join(DATA_FOLDER_PATH, design_problem_input_file_name)

        design_problem.write_needed_inputs(design_problem_inputs)
        design_problem.read_inputs()

        design_problem.model_options["*"] = {
            "cell_capacity_ref": cell_capacity_ref * k_bed,
            "cell_weight_ref": 11.7e-3,
            "reference_curve_current": [100.0, 1000.0, 3000.0, 3720.0],
            "reference_curve_relative_capacity": [1.0, 0.99, 0.98, 0.97],
        }

        design_problem.setup()

        design_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
            val=c_rate_caliber,
            units="h**-1",
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
            val=c_rate_caliber,
            units="h**-1",
        )
        design_problem.set_val(
            "data:TLAR:range",
            val=design_range + 10.0,  # We already now the values at the design point
            units="nmi",
        )
        design_problem.set_val(
            "data:TLAR:NPAX_design",
            val=number_of_pax,
        )
        design_problem.set_val(
            "data:geometry:cabin:seats:passenger:NPAX_max",
            val=number_of_pax,
        )

        # Switching frequencies
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency_mission",
            units="Hz",
            val=switching_frequency,
        )

        # Resistance scaling
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:igbt:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:igbt:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:diode:k_resistance",
            val=resistance_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:igbt:k_resistance",
            val=resistance_scaling,
        )

        # Losses
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:k_igbt_switching_losses",
            val=sw_losses_scaling,
        )
        design_problem.set_val(
            "settings:propulsion:he_power_train:inverter:inverter_1:k_diode_switching_losses",
            val=rr_losses_scaling,
        )

        # Gate voltages
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_2:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:diode:gate_voltage",
            units="V",
            val=gate_voltage_module,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:inverter:inverter_1:igbt:gate_voltage",
            units="V",
            val=gate_voltage_igbt,
        )

        # E-motor parameter
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:tangential_stress",
            units="N/m**2",
            val=tangential_stress,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:design_phase_current_density",
            units="A/m**2",
            val=current_density * 1e6,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:design_surface_current_density",
            units="kA/m",
            val=surface_current_density,
        )
        design_problem.set_val(
            "data:propulsion:he_power_train:SM_PMSM:motor_1:pole_pairs_number", val=nb_pole_pair
        )

        # Run the design problem
        design_problem.run_model()
        design_problem_output_file_name = str(int(investigated_energy_density)) + "_wh_per_kg.xml"
        design_problem.output_file_path = pth.join(
            pth.join(RESULTS_TIME_SENSITIVITY, "design"), design_problem_output_file_name
        )
        print(design_problem.get_val("data:weight:aircraft:MTOW", units="kg")[0])

        # Acceptable range sensitivity was computed here to be 7 kg/km based on the weight of the
        # original Kodiak, so around 13 kg/nm

        # For 395 Wh/kg it is at 16.66.
        # For 500 Wh/kg it is at 10.8, so it is acceptable
        # Sanity check, for 600 Wh/kg it is at 9.0, so it is acceptable


def test_run_lca_time_sensitivity():
    """
    Does the LCA of the test just above. Be careful ! Years in the LCA conf file will need to be
    updated manually !
    """

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    lca_problem_input_file_name = "lca_problem_inputs.xml"
    lca_problem_process_file_name = "lca_problem.yml"

    investigated_energy_densities = np.array([1200] * 1 + [1300] * 2)
    investigated_lifespans = np.array([12000] + [11000, 12000])

    for investigated_energy_density, investigated_lifespan in zip(
        investigated_energy_densities, investigated_lifespans
    ):
        # Like for the other technological parameters, BtF will be assumed to scale based on BED
        buy_to_fly = np.interp(investigated_energy_density, [395, 700, 1301], [7.5, 1.0, 1.0])

        # Now we create the LCA problem
        configurator_lca_problem = oad.FASTOADProblemConfigurator(
            pth.join(DATA_FOLDER_PATH, lca_problem_process_file_name)
        )
        lca_problem = configurator_lca_problem.get_problem()

        design_output_file_path = pth.join(
            pth.join(RESULTS_TIME_SENSITIVITY, "design"),
            str(int(investigated_energy_density)) + "_wh_per_kg.xml",
        )

        # Load inputs
        merge_lca_data_xml(
            design_output_file_path, pth.join(DATA_FOLDER_PATH, lca_problem_input_file_name)
        )
        lca_problem_inputs = design_output_file_path

        lca_problem.write_needed_inputs(lca_problem_inputs)
        lca_problem.read_inputs()

        lca_problem.setup()

        lca_problem.set_val(
            "data:environmental_impact:buy_to_fly:metallic",
            val=buy_to_fly,
        )

        lca_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan",
            val=investigated_lifespan,
        )
        lca_problem.set_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_2:lifespan",
            val=investigated_lifespan,
        )

        lca_problem.run_model()
        file_name = (
            str(int(investigated_energy_density))
            + "_wh_per_kg_"
            + str(int(investigated_lifespan))
            + "_cycles.xml"
        )
        lca_problem.output_file_path = pth.join(
            pth.join(RESULTS_TIME_SENSITIVITY, "final"), file_name
        )
        lca_problem.write_outputs()


def test_preprocess_results():
    """
    Preprocess the results as a csv to facilitate the drawing of the plot in the next function.
    """
    lifespans, beds, single_scores = [], [], []

    for dirpath, _, filenames in os.walk(pth.join(RESULTS_TIME_SENSITIVITY, "final")):
        for filename in filenames:
            datafile = oad.DataFile(pth.join(pth.join(RESULTS_TIME_SENSITIVITY, "final"), filename))
            beds.append(int(filename.split("_wh_per_kg")[0]))
            lifespans.append(
                datafile[
                    "data:propulsion:he_power_train:battery_pack:battery_pack_1:lifespan"
                ].value[0]
            )
            single_scores.append(datafile["data:environmental_impact:single_score"].value[0])

    results_df = pd.DataFrame(columns=["Lifespan", "Battery energy density", "Single score"])
    results_df["Lifespan"] = lifespans
    results_df["Battery energy density"] = beds
    results_df["Single score"] = single_scores
    results_df.to_csv(pth.join(RESULTS_TIME_SENSITIVITY, "recap.csv"))


def test_time_sensitivity_post_processing():
    """
    Post-processes the results obtained with the previous tests.
    """

    results_df = pd.read_csv(pth.join(RESULTS_TIME_SENSITIVITY, "recap.csv"))

    lifespans = results_df["Lifespan"].to_numpy()
    beds = results_df["Battery energy density"].to_numpy()
    single_scores = results_df["Single score"].to_numpy()

    fig = go.Figure()

    # year_continuous = 25 + 1.823088191771652e-05 * (
    #         (395 - energy_densities_grid) ** 2 + 0.0051968503937007875 * (lifespans_grid - 150) ** 2
    # )
    # year_rounded = 2025 + 5 * (
    #         1.823088191771652e-05
    #         * ((395 - energy_densities_grid) ** 2 + 0.0051968503937007875 * (
    #             lifespans_grid - 150) ** 2)
    #         // 5
    # )

    scatter = go.Scatter(
        x=lifespans,
        y=beds,
        marker=dict(color="black", size=5, symbol="cross"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter)

    scatter_2025 = go.Scatter(
        x=[150],
        y=[395],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_2025)
    fig.add_annotation(
        x=150,
        y=395,
        text="2025 w/ Li-Ion",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="left",
    )

    scatter_na_ion = go.Scatter(
        x=[12000],
        y=[700],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_na_ion)
    fig.add_annotation(
        x=12000,
        y=700,
        text="2040 w/ Sodium-Ion",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="right",
    )

    scatter_li_s = go.Scatter(
        x=[1000],
        y=[1300],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_li_s)
    fig.add_annotation(
        x=1000,
        y=1300,
        text="2040 w/ Li-S",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="left",
    )

    # Defining a few constants for the iso-years curves
    k_1 = 0.0051968503937007875
    k_2 = 1.823088191771652e-05

    for year in [2030, 2035, 2040, 2045, 2050]:
        distance_to_ref_year = year - 2025
        x_threshold = np.sqrt(distance_to_ref_year / k_1 / k_2) + 150
        x_for_plot = np.linspace(150, x_threshold, 500)
        y_for_plot = 395 + np.sqrt(distance_to_ref_year / k_2 - k_1 * (x_for_plot - 150) ** 2)
        scatter_year = go.Scatter(
            x=x_for_plot,
            y=y_for_plot,
            line=dict(color="red", dash="dash"),
            mode="lines",
            showlegend=False,
        )
        fig.add_trace(scatter_year)

    fig.add_annotation(
        x=150,
        y=920,
        text="2030",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=150,
        y=1135,
        text="2035",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=7100,
        y=1350,
        text="2045",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=10300,
        y=1350,
        text="2050",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )

    scatter_feasibility = go.Scatter(
        x=[-200, 12350],
        y=[450, 450],
        mode="lines",
        showlegend=False,
        line=dict(color="grey", width=2),
    )
    fig.add_trace(scatter_feasibility)
    fig.add_annotation(
        x=12350,
        y=450,
        text="Feasibility limit",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=15, color="grey", family="Arial Black"),
        align="right",
    )

    # For what we will call the optimal trajectory, we'll look for the immediate biggest gain.
    # If no gains can be reached with immediate neighbours, we expand to the neighbours' neighbours.
    bed_history = [395]
    lifespan_history = [150.0]

    # First step is a necessary improvement of the BED to allow feasibility
    current_bed = 500
    current_lifespan = 150.0

    interp = LinearNDInterpolator(list(zip(lifespans, beds)), single_scores)

    failsafe = 0

    while not (current_bed == 1300 and current_lifespan == 12000.0) and failsafe < 100:
        # We look for next point
        next_point = []
        neighbours = _find_neighbour(current_lifespan, current_bed)

        current_score = interp(current_lifespan, current_bed)

        while not next_point:
            # If any of the neighbour is the final point we exit as this will always be the final
            # point
            if (12000.0, 1300) in neighbours:
                next_point = (12000.0, 1300)
                break

            # First check if there is an improvement in the current score, we check for
            # significant reduction in the single score, with a tolerance of 3%. Otherwise, it's not
            # worth considering
            neighbour_score = []
            for neighbour in neighbours:
                neighbour_score.append(interp(neighbour[0], neighbour[1]))

            # There is a neighbour with significant reduction
            if np.any(np.array(neighbour_score) <= 0.97 * current_score):
                index_best_neighbour = np.argmin(np.array(neighbour_score))
                next_point = neighbours[index_best_neighbour]
                break  # This line might be redundant since the line above exits the while

            else:  # No immediate neighbour with significant gain so we update the list of neighbour with neighbours' neighbour
                new_neighbours = []
                for neighbour in neighbours:
                    neighbours_neighbours = _find_neighbour(neighbour[0], neighbour[1])
                    for neighbours_neighbour in neighbours_neighbours:
                        if neighbours_neighbour not in new_neighbours:
                            new_neighbours.append(neighbours_neighbour)
                neighbours = new_neighbours

        # If we have found a suitable neighbour, we add the current point to history and move
        # to the next point.
        lifespan_history.append(current_lifespan)
        current_lifespan = next_point[0]
        bed_history.append(current_bed)
        current_bed = next_point[1]

        failsafe += 1

    # The loop should exit before we've had a chance to add the final point so we add it manually
    lifespan_history.append(12000)
    bed_history.append(1300)

    heatmap = go.Heatmap(z=single_scores, x=lifespans, y=beds, colorscale="blugrn", zsmooth=False)
    fig.add_trace(heatmap)

    for index, _ in enumerate(lifespan_history):
        scatter_opt_traj = go.Scatter(
            x=lifespan_history[: index + 1],
            y=bed_history[: index + 1],
            line=dict(color="black", width=3),
            marker=dict(color="black", size=10, symbol="diamond"),
            mode="lines+markers",
            showlegend=False,
            name="opt_traj",
        )
        fig.add_trace(scatter_opt_traj)

        fig.update_layout(
            title=None,
            showlegend=True,
            margin=dict(l=5, r=5, t=60, b=5),
            title_font=dict(size=20),
            legend_font=dict(size=20),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            width=1800,
            height=800,
        )
        fig.update_xaxes(title_font=dict(size=15), range=[-200, 12350])
        fig.update_yaxes(title_font=dict(size=15), range=[345, 1350])

        # fig.show()
        fig.write_image(pth.join(pth.join(RESULTS_TIME_SENSITIVITY, "images"), str(index) + ".png"))


def test_time_sensitivity_post_processing_other_strategy():
    """
    Post-processes the results obtained with the previous tests with a different optimization
    strategy.
    """

    results_df = pd.read_csv(pth.join(RESULTS_TIME_SENSITIVITY, "recap.csv"))

    lifespans = results_df["Lifespan"].to_numpy()
    beds = results_df["Battery energy density"].to_numpy()
    single_scores = results_df["Single score"].to_numpy()

    fig = go.Figure()

    # year_continuous = 25 + 1.823088191771652e-05 * (
    #         (395 - energy_densities_grid) ** 2 + 0.0051968503937007875 * (lifespans_grid - 150) ** 2
    # )
    # year_rounded = 2025 + 5 * (
    #         1.823088191771652e-05
    #         * ((395 - energy_densities_grid) ** 2 + 0.0051968503937007875 * (
    #             lifespans_grid - 150) ** 2)
    #         // 5
    # )

    scatter = go.Scatter(
        x=lifespans,
        y=beds,
        marker=dict(color="black", size=5, symbol="cross"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter)

    scatter_2025 = go.Scatter(
        x=[150],
        y=[395],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_2025)
    fig.add_annotation(
        x=150,
        y=395,
        text="2025 w/ Li-Ion",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="left",
    )

    scatter_na_ion = go.Scatter(
        x=[12000],
        y=[700],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_na_ion)
    fig.add_annotation(
        x=12000,
        y=700,
        text="2040 w/ Sodium-Ion",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="right",
    )

    scatter_li_s = go.Scatter(
        x=[1000],
        y=[1300],
        marker=dict(color="red", size=17, symbol="circle"),
        mode="markers",
        showlegend=False,
    )
    fig.add_trace(scatter_li_s)
    fig.add_annotation(
        x=1000,
        y=1300,
        text="2040 w/ Li-S",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=17, color="red", family="Arial Black"),
        align="left",
    )

    # Defining a few constants for the iso-years curves
    k_1 = 0.0051968503937007875
    k_2 = 1.823088191771652e-05

    for year in [2030, 2035, 2040, 2045, 2050]:
        distance_to_ref_year = year - 2025
        x_threshold = np.sqrt(distance_to_ref_year / k_1 / k_2) + 150
        x_for_plot = np.linspace(150, x_threshold, 500)
        y_for_plot = 395 + np.sqrt(distance_to_ref_year / k_2 - k_1 * (x_for_plot - 150) ** 2)
        scatter_year = go.Scatter(
            x=x_for_plot,
            y=y_for_plot,
            line=dict(color="red", dash="dash"),
            mode="lines",
            showlegend=False,
        )
        fig.add_trace(scatter_year)

    fig.add_annotation(
        x=150,
        y=920,
        text="2030",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=150,
        y=1135,
        text="2035",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=7100,
        y=1350,
        text="2045",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )
    fig.add_annotation(
        x=10300,
        y=1350,
        text="2050",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=15, color="red", family="Arial Black"),
        align="left",
    )

    scatter_feasibility = go.Scatter(
        x=[-200, 12350],
        y=[450, 450],
        mode="lines",
        showlegend=False,
        line=dict(color="grey", width=2),
    )
    fig.add_trace(scatter_feasibility)
    fig.add_annotation(
        x=12350,
        y=450,
        text="Feasibility limit",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=15, color="grey", family="Arial Black"),
        align="right",
    )

    # For what we will call the optimal trajectory, we'll look for the immediate biggest gain.
    # If no gains can be reached with immediate neighbours, we expand to the neighbours' neighbours.
    bed_history = [395, 500]
    lifespan_history = [150.0, 150]

    rounded_years = 2025 + 5 * (
        1.823088191771652e-05
        * ((395 - beds) ** 2 + 0.0051968503937007875 * (lifespans - 150) ** 2)
        // 5
    )

    years_investigated = [2030, 2035, 2040, 2045, 2050, 2055]

    for year_investigated in years_investigated:
        suitable_design = np.where(rounded_years <= year_investigated)

        suitable_bed = beds[suitable_design]
        suitable_lifespan = lifespans[suitable_design]

        coherent_design = np.where(suitable_bed >= bed_history[-1]) and np.where(
            suitable_lifespan >= lifespan_history[-1]
        )

        best_design = np.argmin(np.array(single_scores)[suitable_design][coherent_design])
        bed_history.append(suitable_bed[coherent_design][best_design])
        lifespan_history.append(suitable_lifespan[coherent_design][best_design])

    # The loop should exit before we've had a chance to add the final point so we add it manually
    lifespan_history[-1] = 12000
    bed_history[-1] = 1300

    heatmap = go.Heatmap(z=single_scores, x=lifespans, y=beds, colorscale="blugrn", zsmooth=False)
    fig.add_trace(heatmap)

    for index, _ in enumerate(lifespan_history):
        scatter_opt_traj = go.Scatter(
            x=lifespan_history[: index + 1],
            y=bed_history[: index + 1],
            line=dict(color="black", width=3),
            marker=dict(color="black", size=10, symbol="diamond"),
            mode="lines+markers",
            showlegend=False,
            name="opt_traj",
        )
        fig.add_trace(scatter_opt_traj)

        fig.update_layout(
            title=None,
            showlegend=True,
            margin=dict(l=5, r=5, t=60, b=5),
            title_font=dict(size=20),
            legend_font=dict(size=20),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            width=1800,
            height=800,
        )
        fig.update_xaxes(title_font=dict(size=15), range=[-200, 12350])
        fig.update_yaxes(title_font=dict(size=15), range=[345, 1350])

        # fig.show()
        fig.write_image(
            pth.join(pth.join(RESULTS_TIME_SENSITIVITY, "images_v2"), str(index) + ".png")
        )

    fig.show()
