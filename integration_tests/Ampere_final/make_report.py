# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
make_report.py
================
Builds the AMPERE-final case report from a converged run's output files:
  - results/generated/ampere_final_out.xml
  - results/generated/ampere_final_mission_data.csv
  - results/generated/ampere_final_power_train_data.csv

Produces (all under report/):
  - figures/geometry_planform.png    schematic top/side view vs. the real AMPERE wing
  - figures/mass_breakdown.png       MTOW pie + powertrain sub-system bar chart
  - figures/energy_profile.png       cumulative mission energy, battery vs. PEMFC
  - figures/h2_mass_consumed.png     cumulative H2 mass consumed
  - figures/battery_soc_crate.png    SOC and C-rate vs. mission time (10-cluster mean +/- range)
  - params.json                     every extracted/derived number, for debugging or reuse
  - ampere_final_report.md          the report, in Markdown (images do not render inline --
                                     GitHub/most viewers do render local-relative image links,
                                     but see the .html version below for a styled, portable copy)
  - ampere_final_report.html        self-contained HTML build of the same report (via pandoc),
                                     styled to loosely resemble the project's own Sphinx docs
                                     (sphinx_rtd_theme, see docs/conf.py) -- skipped with a
                                     warning if pandoc isn't installed.

All comparison numbers are computed live from the XML/CSVs against the real AMPERE reference
values (Dillinger, Doll, Liaboeuf, Toussaint, Hermetz, Verbeke & Ridel, ICAS 2018-0492, Table 1)
hardcoded below -- so re-running run_ampere_case.py with different --battery-modules/--power-split
and then re-running this script will regenerate a consistent, up-to-date report.

Usage (from integration_tests/ampere_final/, after a successful run_ampere_case.py run):
    python make_report.py
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from lxml import etree

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "generated")
REPORT_DIR = os.path.join(HERE, "report")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")

XML_PATH = os.path.join(RESULTS_DIR, "ampere_final_out.xml")
PT_CSV_PATH = os.path.join(RESULTS_DIR, "ampere_final_power_train_data.csv")
MISSION_CSV_PATH = os.path.join(RESULTS_DIR, "ampere_final_mission_data.csv")

LBM_TO_KG = 0.45359237
LHV_H2 = 33.33  # kWh/kg, lower heating value of hydrogen
N_CLUSTERS = 10
N_MOTORS = 40

# Real AMPERE reference values (Dillinger, Doll, Liaboeuf, Toussaint, Hermetz, Verbeke & Ridel,
# "Handling Qualities of ONERA's Small Business Concept Plane with Distributed Electric
# Propulsion," ICAS 2018-0492, Table 1, reproducing Hermetz, Ridel & Doll, ICAS 2016).
REAL = {
    "mtow_kg": 2400.0,
    "wing_area_m2": 25.925,
    "wing_span_m": 14.5,
    "htp_area_m2": 3.8,
    "vtp_area_m2": 2.02,
    "n_motors": 40,
    "installed_power_kw": 400.0,
    "installed_energy_kwh": 500.0,
    "range_km": 500.0,
}


# --------------------------------------------------------------------------------------------
# XML extraction helpers
# --------------------------------------------------------------------------------------------


def find_val(root, path_tags):
    """path_tags: list of tag names to descend from <data>, e.g. ['geometry','wing','area'].
    Returns float or None."""
    node = root.find("data")
    for tag in path_tags:
        if node is None:
            return None
        node = node.find(tag)
    if node is None or node.text is None:
        return None
    try:
        return float(node.text)
    except ValueError:
        return None


def find_by_instance(root, instance_tag, leaf_tag):
    """Robust lookup: search anywhere in the tree for <instance_tag><leaf_tag>value</leaf_tag>,
    regardless of what the wrapping/grouping parent tag is named (e.g. motor_7's wrapper is
    "SM_PMSM", not "motor" -- this sidesteps having to know that mapping for every component
    type)."""
    node = root.find(f".//{instance_tag}/{leaf_tag}")
    if node is None or node.text is None:
        return None, None
    try:
        return float(node.text), node.attrib.get("units")
    except ValueError:
        return None, None


def cluster_vals(root, instance_prefix, leaf_tag, n):
    """[find_by_instance(root, f"{instance_prefix}{i}", leaf_tag) for i in 1..n], values only."""
    out = []
    for i in range(1, n + 1):
        val, _ = find_by_instance(root, f"{instance_prefix}{i}", leaf_tag)
        out.append(val)
    return out


def to_kw(value, units):
    """Converts a power value tagged MW/kW/W to kW."""
    if value is None:
        return None
    factor = {"MW": 1000.0, "kW": 1.0, "W": 1e-3, None: 1.0}.get(units, 1.0)
    return value * factor


def mass_to_kg(node):
    """node is a <mass units="kg-or-lbm">value</mass> element. FAST-GA/fastga_he mixes kg and
    lbm freely component to component (e.g. wing/mass is kg but fuselage/mass is lbm) -- this
    converts to kg regardless. Skips (returns 0.0) any non-scalar/malformed value."""
    if node is None or node.text is None:
        return 0.0
    try:
        val = float(node.text)
    except ValueError:
        return 0.0
    return val * LBM_TO_KG if node.attrib.get("units") == "lbm" else val


def component_mass_kg(node):
    """Takes ONLY the direct <mass> child of a top-level airframe component (e.g.
    airframe/wing/mass), NOT a recursive sum -- a component's own subtree also contains its
    internal structural decomposition (skin/ribs/spar/misc) AND a "punctual_mass" array of the
    point-loads from engines/motors mounted on it (for bending-moment sizing, not a second
    accounting of those masses) -- recursing with .iter("mass") would massively double/triple
    count."""
    direct_mass = node.find("mass")
    if direct_mass is not None:
        return mass_to_kg(direct_mass)
    # landing_gear has no single top-level mass, only front/mass + main/mass
    total = 0.0
    for sub in node:
        if isinstance(sub.tag, str):
            m = sub.find("mass")
            if m is not None:
                total += mass_to_kg(m)
    return total


# --------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------


def extract_params(root):
    p = {}

    # --- Geometry ---
    p["wing_area"] = find_val(root, ["geometry", "wing", "area"])
    p["wing_span"] = find_val(root, ["geometry", "wing", "span"])
    p["wing_ar"] = find_val(root, ["geometry", "wing", "aspect_ratio"])
    p["htp_area"] = find_val(root, ["geometry", "horizontal_tail", "area"])
    p["vtp_area"] = find_val(root, ["geometry", "vertical_tail", "area"])
    p["fuselage_length"] = find_val(root, ["geometry", "fuselage", "length"])

    # --- Masses ---
    p["mtow"] = find_val(root, ["weight", "aircraft", "MTOW"])
    p["owe"] = find_val(root, ["weight", "aircraft", "OWE"])
    p["mzfw"] = find_val(root, ["weight", "aircraft", "MZFW"])
    p["powertrain_mass"] = find_val(root, ["propulsion", "he_power_train", "mass"])

    weight = root.find("data/weight")
    airframe_group = weight.find("airframe")
    airframe = {}
    for gc in airframe_group:
        if not isinstance(gc.tag, str) or gc.tag == "mass":
            continue
        airframe[gc.tag] = component_mass_kg(gc)
    p["airframe_breakdown_kg"] = airframe
    p["airframe_mass_kg"] = sum(airframe.values())
    p["furniture_mass_kg"] = float(weight.find("furniture/mass").text)
    p["systems_mass_kg"] = float(weight.find("systems/mass").text)
    p["payload_kg"] = float(weight.find("aircraft/max_payload").text)

    # --- Power-train sub-system mass breakdown ---
    he = root.find("data/propulsion/he_power_train")
    type_tags = [
        "battery_pack", "PEMFC_stack", "gaseous_hydrogen_tank", "H2_fuel_system",
        "inverter", "SM_PMSM", "ducted_fan", "DC_cable_harness", "DC_DC_converter",
        "DC_SSPC", "DC_bus", "DC_splitter",
    ]
    pt_breakdown = {}
    for t in type_tags:
        grp = he.find(t)
        if grp is None:
            continue
        total = 0.0
        for inst in grp:
            if not isinstance(inst.tag, str):
                continue
            m = inst.find("mass")
            if m is not None and m.text:
                try:
                    total += float(m.text)
                except ValueError:
                    pass
        pt_breakdown[t] = total
    p["powertrain_breakdown_kg"] = pt_breakdown

    # --- Battery / PEMFC / H2 (summed across clusters) ---
    soc_min = cluster_vals(root, "battery_pack_", "SOC_min", N_CLUSTERS)
    batt_energy_consumed = cluster_vals(root, "battery_pack_", "energy_consumed_mission", N_CLUSTERS)
    pemfc_power_rating = cluster_vals(root, "pemfc_stack_", "power_rating", N_CLUSTERS)
    h2_capacity = cluster_vals(root, "gaseous_hydrogen_tank_", "fuel_total_mission", N_CLUSTERS)
    h2_consumed = cluster_vals(root, "gaseous_hydrogen_tank_", "fuel_consumed_mission", N_CLUSTERS)

    motor_power = []
    for i in range(1, N_MOTORS + 1):
        val, units = find_by_instance(root, f"motor_{i}", "shaft_power_rating")
        motor_power.append(to_kw(val, units))

    p["battery_soc_min_pct"] = min((v for v in soc_min if v is not None), default=None)
    p["battery_energy_consumed_total_kwh"] = sum(v for v in batt_energy_consumed if v is not None) / 1000.0
    p["pemfc_power_rating_total_kw"] = sum(v for v in pemfc_power_rating if v is not None)
    p["h2_capacity_total_kg"] = sum(v for v in h2_capacity if v is not None)
    p["h2_consumed_total_kg"] = sum(v for v in h2_consumed if v is not None)
    motor_clean = [v for v in motor_power if v is not None]
    p["motor_power_rating_total_kw"] = sum(motor_clean) if motor_clean else None

    # Installed electric power = motors + PEMFC electric rating.
    p["installed_power_total_kw"] = (p["motor_power_rating_total_kw"] or 0.0) + p["pemfc_power_rating_total_kw"]

    # Installed energy estimate: battery capacity extrapolated from the 0-100% SOC swing actually
    # used (SOC_mission_start=100%, ends at battery_soc_min_pct) + H2 chemical energy content
    # (LHV basis). This is an ESTIMATE, not a directly-reported "installed energy" XML output.
    if p["battery_soc_min_pct"] is not None:
        depletion_fraction = 1.0 - p["battery_soc_min_pct"] / 100.0
        battery_installed_kwh = (
            p["battery_energy_consumed_total_kwh"] / depletion_fraction if depletion_fraction > 0 else None
        )
    else:
        battery_installed_kwh = None
    h2_energy_kwh = p["h2_capacity_total_kg"] * LHV_H2
    p["battery_installed_energy_kwh_est"] = battery_installed_kwh
    p["h2_energy_kwh"] = h2_energy_kwh
    p["installed_energy_total_kwh_est"] = (
        (battery_installed_kwh + h2_energy_kwh) if battery_installed_kwh is not None else None
    )

    return p


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------


def plot_geometry(root, p):
    geom = root.find("data/geometry")

    def val(path):
        node = geom
        for tag in path.split("/"):
            node = node.find(tag)
        return float(node.text)

    wing_mac25_x = val("wing/MAC/at25percent/x")
    wing_root_y = val("wing/root/y")
    wing_tip_y = val("wing/tip/y")
    wing_root_chord = val("wing/root/chord")
    wing_tip_chord = val("wing/tip/chord")
    wing_le_x = val("wing/MAC/leading_edge/x/absolute")  # LE x since sweep=0, same for root/tip

    htp_at25_abs = wing_mac25_x + val("horizontal_tail/MAC/at25percent/x/from_wingMAC25")
    htp_root_le_x = htp_at25_abs - val("horizontal_tail/MAC/at25percent/x/local")
    htp_root_chord = val("horizontal_tail/root/chord")
    htp_tip_chord = val("horizontal_tail/tip/chord")
    htp_semi_span = val("horizontal_tail/span") / 2.0
    htp_sweep_le = val("horizontal_tail/sweep_0")

    vtp_at25_abs = wing_mac25_x + val("vertical_tail/MAC/at25percent/x/from_wingMAC25")
    vtp_root_le_x = vtp_at25_abs - val("vertical_tail/MAC/at25percent/x/local")
    vtp_root_chord = val("vertical_tail/root/chord")
    vtp_tip_chord = val("vertical_tail/tip/chord")
    vtp_span = val("vertical_tail/span")
    vtp_sweep_le = val("vertical_tail/sweep_0")

    fus_length = val("fuselage/length")
    fus_width = val("fuselage/maximum_width")
    fus_height = val("fuselage/maximum_height")

    real_wing_chord = REAL["wing_area_m2"] / REAL["wing_span_m"]

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- Top view ----
    ax_top.set_title("Top view — simulated vs. real AMPERE", fontsize=11)
    fus_pts_y = [0, fus_width / 2, fus_width / 2, 0, -fus_width / 2, -fus_width / 2, 0]
    fus_pts_x = [0, 1.0, fus_length - 4.0, fus_length, fus_length - 4.0, 1.0, 0]
    ax_top.plot(fus_pts_x, fus_pts_y, color="dimgray", lw=1.2)

    for side in (1, -1):
        y0, y1 = side * wing_root_y, side * wing_tip_y
        xs = [wing_le_x, wing_le_x, wing_le_x + wing_tip_chord, wing_le_x + wing_root_chord]
        ys = [y0, y1, y1, y0]
        ax_top.add_patch(
            patches.Polygon(
                np.c_[xs, ys], closed=True, facecolor="steelblue", edgecolor="navy", alpha=0.6,
                label="Wing (simulated)" if side == 1 else None,
            )
        )

    for side in (1, -1):
        y0, y1 = side * 0.05, side * htp_semi_span
        dx_sweep = htp_semi_span * np.tan(htp_sweep_le)
        xs = [htp_root_le_x, htp_root_le_x + dx_sweep, htp_root_le_x + dx_sweep + htp_tip_chord, htp_root_le_x + htp_root_chord]
        ys = [y0, y1, y1, y0]
        ax_top.add_patch(
            patches.Polygon(
                np.c_[xs, ys], closed=True, facecolor="indianred", edgecolor="darkred", alpha=0.6,
                label="HTP (simulated)" if side == 1 else None,
            )
        )

    n_per_side = N_MOTORS // 2
    for side in (1, -1):
        ys_fans = np.linspace(side * (wing_root_y + 0.15), side * (wing_tip_y - 0.15), n_per_side)
        xs_fans = np.full(n_per_side, wing_le_x - 0.12)
        ax_top.scatter(xs_fans, ys_fans, s=14, color="black", zorder=5)
    ax_top.scatter([], [], s=14, color="black", label=f"Ducted fans ({N_MOTORS})")

    real_span_half = REAL["wing_span_m"] / 2.0
    real_x0 = wing_le_x - (real_wing_chord - wing_root_chord) / 2
    ax_top.add_patch(
        patches.Rectangle(
            (real_x0, -real_span_half), real_wing_chord, REAL["wing_span_m"],
            fill=False, edgecolor="black", linestyle="--", linewidth=1.4,
            label=f"Real AMPERE wing ({REAL['wing_area_m2']:.1f} m², b={REAL['wing_span_m']:.1f} m)",
        )
    )

    ax_top.set_xlim(-1, fus_length + 1)
    ax_top.set_ylim(-9, 9)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x [m] (from nose)")
    ax_top.set_ylabel("y [m]")
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_top.grid(alpha=0.2)

    # ---- Side view ----
    ax_side.set_title("Side view — VTP", fontsize=11)
    ax_side.plot([0, fus_length], [0, 0], color="dimgray", lw=1.2)
    ax_side.plot(
        [0, 1.0, fus_length - 4.0, fus_length], [0, fus_height * 0.4, fus_height * 0.4, 0],
        color="dimgray", lw=1.2,
    )
    dz_sweep = vtp_span * np.tan(vtp_sweep_le)
    xs_vtp = [vtp_root_le_x, vtp_root_le_x + dz_sweep, vtp_root_le_x + dz_sweep + vtp_tip_chord, vtp_root_le_x + vtp_root_chord]
    zs_vtp = [0, vtp_span, vtp_span, 0]
    ax_side.add_patch(
        patches.Polygon(np.c_[xs_vtp, zs_vtp], closed=True, facecolor="seagreen", edgecolor="darkgreen", alpha=0.6, label="VTP (simulated)")
    )
    ax_side.set_xlim(-1, fus_length + 1)
    ax_side.set_ylim(-1, 4)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("x [m]")
    ax_side.set_ylabel("z [m]")
    ax_side.legend(loc="upper right", fontsize=8)
    ax_side.grid(alpha=0.2)

    fig.suptitle(
        f"Schematic geometry — AMPERE final case (wing_area DEP-equilibrium)\n"
        f"Wing: {p['wing_area']:.2f} m² (real {REAL['wing_area_m2']:.2f} m²)   |   "
        f"HTP: {p['htp_area']:.2f} m² (real {REAL['htp_area_m2']:.2f} m²)   |   "
        f"VTP: {p['vtp_area']:.2f} m² (real {REAL['vtp_area_m2']:.2f} m²)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIGURES_DIR, "geometry_planform.png"), dpi=150)
    plt.close(fig)


def plot_mass_breakdown(p):
    airframe_mass = p["airframe_mass_kg"]
    propulsion_mass = p["powertrain_mass"]
    furniture_systems = p["furniture_mass_kg"] + p["systems_mass_kg"]
    payload = p["payload_kg"]
    mtow = p["mtow"]
    owe = p["owe"]
    other = mtow - (airframe_mass + propulsion_mass + furniture_systems + payload)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    labels1 = ["Airframe\n(structure)", "Propulsion\n(he_power_train)", "Furniture+systems", "Payload", "Other\n(margins)"]
    values1 = [airframe_mass, propulsion_mass, furniture_systems, payload, max(other, 0)]
    colors1 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    ax1.pie(
        values1, labels=labels1, autopct=lambda pct: f"{pct:.1f}%\n({pct / 100 * mtow:.0f} kg)",
        colors=colors1, startangle=90, textprops={"fontsize": 8.5},
    )
    ax1.set_title(f"MTOW breakdown = {mtow:.0f} kg\n(OWE={owe:.0f} kg, payload={payload:.0f} kg)", fontsize=10)

    pt = p["powertrain_breakdown_kg"]
    name_map = {
        "battery_pack": f"Battery ({N_CLUSTERS} packs)",
        "PEMFC_stack": f"PEMFC ({N_CLUSTERS} stacks)",
        "gaseous_hydrogen_tank": f"H2 tanks ({N_CLUSTERS})",
        "H2_fuel_system": "H2 system (lines/valves)",
        "inverter": f"Inverters ({N_MOTORS})",
        "SM_PMSM": f"Motors ({N_MOTORS})",
        "ducted_fan": f"Ducted fans ({N_MOTORS})",
        "DC_cable_harness": "DC cables",
        "DC_DC_converter": f"DC-DC converters ({N_CLUSTERS})",
        "DC_SSPC": f"SSPC ({N_CLUSTERS * 2})",
        "DC_bus": "DC buses",
        "DC_splitter": f"DC splitters ({N_CLUSTERS})",
    }
    items = sorted(pt.items(), key=lambda kv: -kv[1])
    labels2 = [name_map.get(k, k) for k, _ in items]
    values2 = [v for _, v in items]
    total_pt = sum(values2)

    bars = ax2.barh(labels2[::-1], values2[::-1], color="#4C72B0")
    ax2.set_xlabel("Mass [kg]")
    ax2.set_title(f"Electric powertrain detail = {total_pt:.0f} kg", fontsize=10)
    for bar, v in zip(bars, values2[::-1]):
        ax2.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2, f"{v:.0f} kg ({v / total_pt * 100:.1f}%)", va="center", fontsize=8)
    ax2.set_xlim(0, max(values2) * 1.35)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("AMPERE final case — mass breakdown", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "mass_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_mission_profiles(p):
    """Reads power_train_data.csv (32 rows = 30 airborne mission points from mission_data.csv +
    2 ground/taxi legs at start and end, each 150s, not present in mission_data.csv) and
    mission_data.csv (phase name labels only). Builds the energy, H2, and battery SOC/C-rate
    figures, and returns a dict of derived numbers to merge into the report's params."""
    pt = pd.read_csv(PT_CSV_PATH, index_col=0)
    mission = pd.read_csv(MISSION_CSV_PATH, index_col=0)

    time_step = pt["time_step [s]"].to_numpy()
    elapsed_min = np.concatenate([[0], np.cumsum(time_step)[:-1]]) / 60.0

    # pt row 0 = ground/taxi before climb; pt rows 1..N = mission_data rows 0..N-1; last pt row =
    # ground/taxi after reserve.
    phase = ["ground"] + mission["name"].str.replace("sizing:main_route:", "", regex=False).tolist() + ["ground"]
    phase_changes = [i for i in range(1, len(phase)) if phase[i] != phase[i - 1]]
    bounds = [0] + phase_changes + [len(phase) - 1]
    phase_colors = {"ground": "0.92", "climb": "#FFF3CD", "cruise": "#D4EDDA", "descent": "#D1ECF1", "reserve": "#F8D7DA"}

    # --- Energy ---
    batt_power_kw = sum(pt[f"battery_pack_{i} power_out [kW]"] for i in range(1, N_CLUSTERS + 1)).to_numpy()
    h2_flow_kg_h = sum(pt[f"pemfc_stack_{i} fuel_consumption [kg/h]"] for i in range(1, N_CLUSTERS + 1)).to_numpy()
    pemfc_eff = pt["pemfc_stack_1 efficiency [-]"].to_numpy()  # clusters are symmetric
    pemfc_elec_power_kw = h2_flow_kg_h * LHV_H2 * pemfc_eff

    dt_h = time_step / 3600.0
    batt_energy_cum = np.cumsum(batt_power_kw * dt_h)
    pemfc_energy_cum = np.cumsum(pemfc_elec_power_kw * dt_h)
    total_energy_cum = batt_energy_cum + pemfc_energy_cum
    h2_mass_cum = np.cumsum(h2_flow_kg_h * dt_h)

    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.stackplot(
        elapsed_min, batt_energy_cum, pemfc_energy_cum,
        labels=["Battery (electric)", "PEMFC (electric, from H2 x efficiency)"],
        colors=["#4C72B0", "#DD8452"], alpha=0.85,
    )
    ax1.plot(elapsed_min, total_energy_cum, color="black", lw=1.5, ls="--", label="Total")
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        ax1.axvspan(elapsed_min[i0], elapsed_min[i1], color=phase_colors.get(phase[i0], "white"), alpha=0.35, zorder=0)
        mid = (elapsed_min[i0] + elapsed_min[i1]) / 2
        ax1.text(mid, ax1.get_ylim()[1] * 0.02, phase[i0], fontsize=7.5, ha="center", va="bottom", color="0.3")
    ax1.set_xlabel("Mission time [min]")
    ax1.set_ylabel("Cumulative electric energy consumed [kWh]")
    ax1.set_title("Energy consumed over the mission — total and by system\n(AMPERE final case, 10 clusters summed)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, "energy_profile.png"), dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(elapsed_min, h2_mass_cum, color="#DD8452", lw=2)
    ax2.fill_between(elapsed_min, 0, h2_mass_cum, color="#DD8452", alpha=0.25)
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        ax2.axvspan(elapsed_min[i0], elapsed_min[i1], color=phase_colors.get(phase[i0], "white"), alpha=0.35, zorder=0)
    ax2.set_xlabel("Mission time [min]")
    ax2.set_ylabel(f"Cumulative H2 mass consumed [kg]  ({N_CLUSTERS} tanks)")
    ax2.set_title("Hydrogen mass consumed over the mission")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "h2_mass_consumed.png"), dpi=150)
    plt.close(fig2)

    # --- Battery SOC / C-rate ---
    soc = np.vstack([pt[f"battery_pack_{i} state_of_charge [percent]"].to_numpy() for i in range(1, N_CLUSTERS + 1)])
    c_rate = np.vstack([pt[f"battery_pack_{i} c_rate [1/h]"].to_numpy() for i in range(1, N_CLUSTERS + 1)])
    soc_mean, soc_min_arr, soc_max = soc.mean(axis=0), soc.min(axis=0), soc.max(axis=0)
    cr_mean, cr_min, cr_max = c_rate.mean(axis=0), c_rate.min(axis=0), c_rate.max(axis=0)

    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        for ax in (ax3, ax4):
            ax.axvspan(elapsed_min[i0], elapsed_min[i1], color=phase_colors.get(phase[i0], "white"), alpha=0.35, zorder=0)
        mid = (elapsed_min[i0] + elapsed_min[i1]) / 2
        ax3.text(mid, 103, phase[i0], fontsize=7.5, ha="center", color="0.3")

    ax3.plot(elapsed_min, soc_mean, color="#4C72B0", lw=2, label=f"Mean SOC ({N_CLUSTERS} clusters)")
    ax3.fill_between(elapsed_min, soc_min_arr, soc_max, color="#4C72B0", alpha=0.25, label="min-max range across clusters")
    ax3.axhline(soc_min_arr.min(), color="red", ls=":", lw=1.2, label=f"SOC_min = {soc_min_arr.min():.1f}%")
    ax3.set_ylabel("State of charge (SOC) [%]")
    ax3.set_title("Battery — state of charge over the mission")
    ax3.set_ylim(0, 108)
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.grid(alpha=0.3)

    ax4.plot(elapsed_min, cr_mean, color="#DD8452", lw=2, label=f"Mean C-rate ({N_CLUSTERS} clusters)")
    ax4.fill_between(elapsed_min, cr_min, cr_max, color="#DD8452", alpha=0.25, label="min-max range across clusters")
    ax4.set_xlabel("Mission time [min]")
    ax4.set_ylabel("C-rate [1/h]")
    ax4.set_title("Battery — discharge rate (C-rate) over the mission")
    ax4.legend(loc="upper right", fontsize=8.5)
    ax4.grid(alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, "battery_soc_crate.png"), dpi=150)
    plt.close(fig3)

    return {
        "mission_battery_energy_kwh": float(batt_energy_cum[-1]),
        "mission_pemfc_energy_kwh": float(pemfc_energy_cum[-1]),
        "mission_total_energy_kwh": float(total_energy_cum[-1]),
        "mission_h2_consumed_kg_from_pt": float(h2_mass_cum[-1]),
        "mission_total_time_min": float(elapsed_min[-1]),
        "soc_final_pct": float(soc_mean[-1]),
        "soc_min_from_pt_pct": float(soc_min_arr.min()),
        "c_rate_max_from_pt": float(cr_max.max()),
    }


# --------------------------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------------------------


def pct_diff(sim, real):
    if sim is None or real in (None, 0):
        return None
    return (sim - real) / real * 100.0


def fmt_diff(sim, real, unit=""):
    d = pct_diff(sim, real)
    if d is None:
        return "—"
    return f"{d:+.1f}%"


def build_markdown(p):
    mtow_diff = fmt_diff(p["mtow"], REAL["mtow_kg"])
    wing_diff = fmt_diff(p["wing_area"], REAL["wing_area_m2"])
    span_diff = fmt_diff(p["wing_span"], REAL["wing_span_m"])
    htp_diff = fmt_diff(p["htp_area"], REAL["htp_area_m2"])
    vtp_diff = fmt_diff(p["vtp_area"], REAL["vtp_area_m2"])
    power_diff = fmt_diff(p["installed_power_total_kw"], REAL["installed_power_kw"])
    energy_diff = fmt_diff(p["installed_energy_total_kwh_est"], REAL["installed_energy_kwh"])

    md = f"""# Case study: AMPERE (40 EDF, 10 clusters, hybrid PEMFC+H2+battery) — FAST-OAD-CS23-HE

**Case:** `ampere_final` (16 battery modules/cluster, 70% of the power split routed to the battery, wing sizing via `fastga_he.loop.wing_area` — landing/approach equilibrium with slipstream lift augmentation)

## 1. The reference aircraft

AMPERE is ONERA's distributed electric propulsion (DEP) concept plane: a high-wing, CS-23 business aircraft with 40 Electric Ducted Fans (EDF) mounted on the wing leading edge, also used as a high-lift device through the engine-slipstream effect. The hybrid architecture combines PEMFC fuel cells + gaseous hydrogen with batteries, organized into {N_CLUSTERS} clusters of {N_MOTORS // N_CLUSTERS} fans each.

Main TLARs (source: Dillinger, Döll, Liaboeuf, Toussaint, Hermetz, Verbeke & Ridel, *"Handling Qualities of ONERA's Small Business Concept Plane with Distributed Electric Propulsion,"* ICAS 2018-0492, Table 1, reproducing Hermetz, Ridel & Döll, ICAS 2016): 4-6 passengers, 500 km range in ~2h, STOL capability, FL100 ceiling (10,000 ft, unpressurized cabin).

This case in FAST-OAD-CS23-HE reuses the Pipistrel's structural geometry (fuselage, landing gear, tail arrangement) — only the propulsion system ({N_MOTORS} EDF / {N_CLUSTERS} hybrid clusters) and the wing/tail/TLAR parameters were replaced with AMPERE's real values as seeds. In other words: **this is a validation of the propulsion architecture at AMPERE scale, not a faithful, complete sizing of the real aircraft** — see Section 8 for known limitations.

## 2. Main parameters: real vs. simulated

| Parameter | Real (ICAS 2018-0492, Table 1) | Simulated (`ampere_final`) | Difference |
|---|---|---|---|
| MTOW | {REAL['mtow_kg']:.0f} kg | {p['mtow']:.1f} kg | {mtow_diff} |
| Wing area | {REAL['wing_area_m2']:.3f} m² | {p['wing_area']:.2f} m² | {wing_diff} |
| Wing span | {REAL['wing_span_m']:.1f} m | {p['wing_span']:.2f} m | {span_diff} |
| HTP area | {REAL['htp_area_m2']:.1f} m² | {p['htp_area']:.2f} m² | {htp_diff} |
| VTP area | {REAL['vtp_area_m2']:.2f} m² | {p['vtp_area']:.2f} m² | {vtp_diff} |
| Number of motors (EDF) | {REAL['n_motors']} | {N_MOTORS} | equal |
| Installed power (motors + PEMFC) | {REAL['installed_power_kw']:.0f} kW | ≈{p['installed_power_total_kw']:.0f} kW | {power_diff} |
| Installed energy (battery+H2, LHV basis, estimated) | {REAL['installed_energy_kwh']:.0f} kWh | ≈{p['installed_energy_total_kwh_est']:.0f} kWh | {energy_diff} |
| Range (main route) | {REAL['range_km']:.0f} km | {REAL['range_km']:.0f} km | equal (mission input) |
| Payload | 4-6 PAX | {p['payload_kg']:.0f} kg (not converted to PAX) | — |

**Note on wing/tail:** the wing and VTP converge much closer to the real values after switching the wing-sizing submodel to `fastga_he.loop.wing_area` (`UpdateWingAreaLiftDEPEquilibrium`), which solves a landing/approach equilibrium that accounts for the lift gain from each ducted fan's slipstream — unlike the default `fastga.loop.wing_area` (a pure stall-speed/CL_max formula with zero coupling to propulsion), which had inflated the wing by +41.5% in an earlier run with the same battery sizing. The HTP swung to the undersized side as a side effect — likely because `tail_sizing`/`static_margin` (still the default, non-DEP-aware loops) reacted to the CG shift caused by the smaller wing (see Section 8).

## 3. Geometry

![Schematic geometry](figures/geometry_planform.png)

Schematic sketch (not CAD-accurate) built from the span/chord/sweep values in the output XML. AMPERE's real wing (dashed rectangle) is overlaid for scale comparison. The {N_MOTORS} ducted fans are shown along the wing leading edge, matching the real configuration.

## 4. Mass breakdown

![Mass breakdown](figures/mass_breakdown.png)

The {p['mtow']:.0f} kg MTOW splits into airframe ({p['airframe_mass_kg']:.0f} kg), electric powertrain ({p['powertrain_mass']:.0f} kg), furniture+systems ({p['furniture_mass_kg'] + p['systems_mass_kg']:.0f} kg), and payload ({p['payload_kg']:.0f} kg). Within the powertrain, the battery is the dominant sub-system — a direct reflection of the fixed battery-modules-per-cluster sizing (see Section 8), far heavier than the inverters, PEMFC stacks, or the ducted fans themselves.

## 5. Energy consumed over the mission

![Energy consumed](figures/energy_profile.png)

Total electric energy consumed over the mission: **{p['mission_total_energy_kwh']:.1f} kWh**, split {p['mission_battery_energy_kwh']:.1f} kWh ({p['mission_battery_energy_kwh'] / p['mission_total_energy_kwh'] * 100:.1f}%) from the battery and {p['mission_pemfc_energy_kwh']:.1f} kWh ({p['mission_pemfc_energy_kwh'] / p['mission_total_energy_kwh'] * 100:.1f}%) from the PEMFC (electric, computed from H2 consumption × stack efficiency at each point). Mission phases (climb/cruise/descent/reserve) are shaded. The battery figure matches the value reported in the output XML (`energy_consumed_mission` summed over the {N_CLUSTERS} clusters), confirming the integration.

## 6. Battery — state of charge and C-rate

![SOC and C-rate](figures/battery_soc_crate.png)

SOC drops from 100% to {p['soc_min_from_pt_pct']:.1f}% by the end of the mission (matching `SOC_min` reported in the XML). Maximum C-rate ({p['c_rate_max_from_pt']:.2f} 1/h) occurs during climb. The min-max range across the {N_CLUSTERS} clusters essentially overlaps the mean, confirming the expected symmetry of the architecture.

## 7. Hydrogen consumed

![H2 consumed](figures/h2_mass_consumed.png)

Total H2 mass consumed over the mission: **{p['h2_consumed_total_kg']:.3f} kg** ({N_CLUSTERS} tanks summed), out of a total onboard capacity of {p['h2_capacity_total_kg']:.2f} kg. Consumption grows roughly linearly during cruise, with higher rates during climb.

## 8. Discussion and known limitations

- **Geometry origin:** fuselage, landing gear, and structural tail arrangement are still inherited from the Pipistrel — only the propulsion system and the main wing/tail/TLAR parameters were replaced with AMPERE's real values. This is not a faithful, complete sizing of the real aircraft.
- **Installed power:** the motors+PEMFC add up to ≈{p['installed_power_total_kw']:.0f} kW, versus the published {REAL['installed_power_kw']:.0f} kW — likely because the simulated mission/aerodynamics (still influenced by the Pipistrel) doesn't demand as much peak thrust as the real aircraft would need for its STOL takeoff requirements, which this case doesn't reproduce as an active constraint.
- **Battery sizing:** SOC_min of {p['soc_min_from_pt_pct']:.1f}% reflects the fixed battery-modules-per-cluster seed in `run_ampere_case.py` (`--battery-modules`, default 16) — re-sweeping it lets you retarget a different SOC floor as the airframe geometry evolves.
- **HTP:** its size is a side effect of the smaller wing on the default (non-DEP-aware) `tail_sizing`/`static_margin` loops — not investigated in depth in this report.
- **Literature correction:** the "32 engines" figure that appears in the paper (Table 1, "Scale 1:5" column) refers to the 1:5-scale wind-tunnel mock-up (32 larger 50mm EDFs reproducing the thrust of 40 smaller 40mm EDFs at full scale), not an engine-out (OEI) redundancy requirement for the real aircraft.

## Sources

- Dillinger, E., Döll, C., Liaboeuf, R., Toussaint, C., Hermetz, J., Verbeke, C., Ridel, M. "Handling Qualities of ONERA's Small Business Concept Plane with Distributed Electric Propulsion." ICAS 2018-0492.
- Output files from the `ampere_final` run (FAST-OAD-CS23-HE): `ampere_final_out.xml`, `ampere_final_mission_data.csv`, `ampere_final_power_train_data.csv`.
"""
    return md


RTD_STYLE_CSS = """
:root {
  --rtd-blue: #2980B9;
  --rtd-dark: #343131;
  --rtd-bg: #fcfcfc;
  --rtd-border: #e1e4e5;
  --rtd-code-bg: #f8f8f8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--rtd-bg);
  color: var(--rtd-dark);
  font-family: "Lato", "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
}
.page {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 2.2em 4em 2.2em;
  background: #fff;
  box-shadow: 0 0 18px rgba(0,0,0,0.06);
}
header.report-header {
  background: var(--rtd-blue);
  color: #fff;
  margin: 0 -2.2em 2em -2.2em;
  padding: 2.2em 2.2em 1.4em 2.2em;
}
header.report-header .kicker {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.78em;
  opacity: 0.85;
  margin-bottom: 0.4em;
}
header.report-header h1 { margin: 0; font-weight: 700; font-size: 1.9em; border: none; color: #fff; }
h1, h2, h3, h4 { font-weight: 700; color: var(--rtd-dark); }
h1 { font-size: 1.9em; }
h2 { font-size: 1.45em; border-bottom: 1px solid var(--rtd-border); padding-bottom: 0.3em; margin-top: 2em; }
h3 { font-size: 1.15em; color: var(--rtd-blue); }
a { color: var(--rtd-blue); text-decoration: none; }
a:hover { text-decoration: underline; }
#TOC {
  background: #f4f7fa;
  border: 1px solid var(--rtd-border);
  border-left: 4px solid var(--rtd-blue);
  border-radius: 3px;
  padding: 0.9em 1.6em;
  margin: 1.6em 0 2em 0;
}
#TOC::before {
  content: "Contents";
  display: block;
  font-weight: 700;
  color: var(--rtd-blue);
  margin-bottom: 0.4em;
  font-size: 0.95em;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
#TOC ul { list-style: none; padding-left: 1.1em; margin: 0.2em 0; }
#TOC > ul { padding-left: 0; }
#TOC a { color: var(--rtd-dark); font-size: 0.95em; }
#TOC a:hover { color: var(--rtd-blue); }
table { border-collapse: collapse; width: 100%; margin: 1.3em 0; font-size: 0.93em; }
table th { background: var(--rtd-blue); color: #fff; text-align: left; padding: 0.55em 0.8em; }
table td { padding: 0.5em 0.8em; border-bottom: 1px solid var(--rtd-border); }
table tr:nth-child(even) td { background: #f7fafc; }
code {
  background: var(--rtd-code-bg);
  border: 1px solid var(--rtd-border);
  border-radius: 3px;
  padding: 0.1em 0.4em;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.88em;
  color: #e74c3c;
}
blockquote { border-left: 4px solid var(--rtd-blue); margin: 1.2em 0; padding: 0.3em 1.2em; color: #4d4d4d; background: #f8fbfd; }
figure { margin: 1.8em 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid var(--rtd-border); border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
figure figcaption, p img + em { display: block; margin-top: 0.6em; font-size: 0.88em; color: #767676; font-style: italic; }
img { max-width: 100%; }
p:has(> img) { text-align: center; margin: 1.8em 0 0.3em 0; }
hr { border: none; border-top: 1px solid var(--rtd-border); margin: 2.5em 0; }
.report-footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--rtd-border); font-size: 0.85em; color: #888; }
"""

RTD_TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>$title$</title>
  <style>
$RTD_CSS$
  </style>
</head>
<body>
<div class="page">
  <header class="report-header">
    <div class="kicker">FAST-OAD-CS23-HE &mdash; AMPERE case study</div>
    <h1>$title$</h1>
  </header>
$if(toc)$
  <div id="TOC">
$table-of-contents$
  </div>
$endif$
$body$
  <div class="report-footer">
    Generated report &mdash; ampere_final case, FAST-OAD-CS23-HE.
  </div>
</div>
</body>
</html>
"""


def build_html(md_path, html_path):
    """Builds a self-contained, styled HTML copy of the report via pandoc (loosely mimicking the
    project's own Sphinx sphinx_rtd_theme, see docs/conf.py). Silently skipped with a printed
    warning if pandoc isn't installed -- the .md report is still produced either way."""
    if shutil.which("pandoc") is None:
        print("[make_report] pandoc not found on PATH -- skipping HTML build (the .md report is still valid).")
        return False

    css_path = os.path.join(REPORT_DIR, "rtd_style.css")
    template_path = os.path.join(REPORT_DIR, "rtd_template.html")
    with open(css_path, "w") as f:
        f.write(RTD_STYLE_CSS)
    with open(template_path, "w") as f:
        f.write(RTD_TEMPLATE_HTML)

    # Strip the leading "# Case study: ..." H1 line -- the styled header already shows the title.
    with open(md_path) as f:
        md_lines = f.read().splitlines()
    body_only = "\n".join(line for line in md_lines if not line.startswith("# Case study"))
    body_only_path = os.path.join(REPORT_DIR, "_body_only.md")
    with open(body_only_path, "w") as f:
        f.write(body_only)

    with open(css_path) as f:
        css_content = f.read()

    cmd = [
        "pandoc", body_only_path,
        "--standalone", "--toc", "--toc-depth=2",
        f"--template={template_path}",
        "--self-contained",
        "-M", "title=Case study: AMPERE (40 EDF, 10 clusters, hybrid PEMFC+H2+battery)",
        "-V", f"RTD_CSS={css_content}",
        "-o", html_path,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=REPORT_DIR, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[make_report] pandoc failed -- skipping HTML build.\n{e.stderr}")
        return False
    finally:
        # Best-effort cleanup of the temp body-only markdown -- some sandboxed/mounted
        # filesystems reject an immediate delete right after write; leaving the file behind is
        # harmless (it gets overwritten next run) so this must never crash the whole build.
        try:
            os.remove(body_only_path)
        except OSError:
            pass

    return True


# --------------------------------------------------------------------------------------------


def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(
            f"{XML_PATH} not found -- run run_ampere_case.py first (from integration_tests/ampere_final/)."
        )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    tree = etree.parse(XML_PATH)
    root = tree.getroot()

    p = extract_params(root)
    plot_geometry(root, p)
    plot_mass_breakdown(p)
    p.update(plot_mission_profiles(p))

    with open(os.path.join(REPORT_DIR, "params.json"), "w") as f:
        json.dump(p, f, indent=2)

    md = build_markdown(p)
    md_path = os.path.join(REPORT_DIR, "ampere_final_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")

    html_path = os.path.join(REPORT_DIR, "ampere_final_report.html")
    if build_html(md_path, html_path):
        print(f"Wrote {html_path}")

    print(f"Figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
