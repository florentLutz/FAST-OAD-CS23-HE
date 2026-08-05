# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
#
# Regenerates the ONERA Ampere power train assembly YAML for the final/promoted case
# (data/generated/assembly_ampere_final.yml): 40 EDF grouped in 10 independent clusters of 4 fans
# (2x2), each cluster with its own PEMFC + gaseous hydrogen tank + battery hybrid source (battery
# covers the "dynamic"/instantaneous share of the load, PEMFC covers the base load), matching the
# redundant, cluster-isolated architecture described in Hermetz, Ridel & Doll (2016 ICAS) for the
# "high-wing, EDF at wing leading edge" Ampere concept-plane (MTOW = 2400 kg, 40 EDF, 400 kW
# installed power).
#
# Structure of each cluster N (N = 1..10), fan indices 4*(N-1)+1 .. 4*N:
#   ducted_fan_i -> motor_i (sm_pmsm) -> inverter_i -> dc_bus_i -> harness_i -> dc_bus_group_N
#   dc_bus_group_N -> harness_central_N -> dc_splitter_N
#     -> dc_sspc_N1 -> battery_pack_N            (battery branch)
#     -> dc_sspc_N2 -> dc_dc_converter_N -> pemfc_stack_N -> h2_fuel_system_N -> gaseous_hydrogen_tank_N

import argparse

N_FANS_PER_CLUSTER = 4
N_CLUSTERS = 10


def generate_assembly_yaml(n_clusters: int = N_CLUSTERS, fans_per_cluster: int = N_FANS_PER_CLUSTER) -> str:
    n_total_fans = n_clusters * fans_per_cluster

    lines = []
    lines.append(
        "title: Generated ONERA Ampere assembly (FINAL case) -- N_FANS_TOTAL={}, N_CLUSTERS={} "
        "(GROUP_SIZE={}), hybrid PEMFC+H2+battery per cluster, NO NOSE ENGINE "
        "(generate_ampere_case.py)".format(n_total_fans, n_clusters, fans_per_cluster)
    )
    lines.append("power_train_components:")

    fan_id = 0
    for cluster in range(1, n_clusters + 1):
        lines.append(f"  # ===== Cluster {cluster} ({fans_per_cluster} EDF + hybrid PEMFC/H2/battery source) =====")
        lines.append(f"  dc_bus_group_{cluster}:")
        lines.append("    id: fastga_he.pt_component.dc_bus")
        lines.append("    options:")
        lines.append("      number_of_inputs: 1")
        lines.append(f"      number_of_outputs: {fans_per_cluster}")
        lines.append("    position: inside_the_wing")

        lines.append(f"  harness_central_{cluster}:")
        lines.append("    id: fastga_he.pt_component.dc_line")
        lines.append("    position: from_rear_to_front")

        lines.append(f"  dc_splitter_{cluster}:")
        lines.append("    id: fastga_he.pt_component.dc_splitter")
        lines.append("    position: in_the_back")

        lines.append(f"  dc_sspc_{cluster}1:")
        lines.append("    id: fastga_he.pt_component.dc_sspc")
        lines.append("    options:")
        lines.append("      closed_by_default: true")
        lines.append("    position: in_the_front")

        lines.append(f"  battery_pack_{cluster}:")
        lines.append("    id: fastga_he.pt_component.battery_pack")
        lines.append("    position: in_the_front")

        lines.append(f"  dc_sspc_{cluster}2:")
        lines.append("    id: fastga_he.pt_component.dc_sspc")
        lines.append("    options:")
        lines.append("      closed_by_default: true")
        lines.append("    position: in_the_back")

        lines.append(f"  dc_dc_converter_{cluster}:")
        lines.append("    id: fastga_he.pt_component.dc_dc_converter")
        lines.append("    position: in_the_back")

        lines.append(f"  pemfc_stack_{cluster}:")
        lines.append("    id: fastga_he.pt_component.pemfc_stack")
        lines.append("    position: in_the_back")

        lines.append(f"  h2_fuel_system_{cluster}:")
        lines.append("    id: fastga_he.pt_component.h2_fuel_system")
        lines.append("    options:")
        lines.append("      number_of_tanks: 1")
        lines.append("      number_of_power_sources: 1")
        lines.append("      wing_related: false")
        lines.append("      compact: false")
        lines.append("    position: in_the_rear")

        lines.append(f"  gaseous_hydrogen_tank_{cluster}:")
        lines.append("    id: fastga_he.pt_component.gaseous_hydrogen_tank")
        lines.append("    position: in_the_cabin")

        for _ in range(fans_per_cluster):
            fan_id += 1
            lines.append(f"  ducted_fan_{fan_id}:")
            lines.append("    id: fastga_he.pt_component.ducted_fan")
            lines.append("    position: on_the_wing")
            lines.append(f"  motor_{fan_id}:")
            lines.append("    id: fastga_he.pt_component.sm_pmsm")
            lines.append("    position: on_the_wing")
            lines.append(f"  inverter_{fan_id}:")
            lines.append("    id: fastga_he.pt_component.inverter")
            lines.append("    position: inside_the_wing")
            lines.append(f"  dc_bus_{fan_id}:")
            lines.append("    id: fastga_he.pt_component.dc_bus")
            lines.append("    options:")
            lines.append("      number_of_inputs: 1")
            lines.append("      number_of_outputs: 1")
            lines.append("    position: inside_the_wing")
            lines.append(f"  harness_{fan_id}:")
            lines.append("    id: fastga_he.pt_component.dc_line")
            lines.append("    position: from_wing_to_nose")

    lines.append("component_connections:")

    fan_id = 0
    for cluster in range(1, n_clusters + 1):
        for k in range(1, fans_per_cluster + 1):
            fan_id += 1
            lines.append(f"  - source: ducted_fan_{fan_id}")
            lines.append(f"    target: motor_{fan_id}")
            lines.append(f"  - source: motor_{fan_id}")
            lines.append(f"    target: inverter_{fan_id}")
            lines.append(f"  - source: inverter_{fan_id}")
            lines.append(f"    target: [dc_bus_{fan_id}, 1]")
            lines.append(f"  - source: [dc_bus_{fan_id}, 1]")
            lines.append(f"    target: harness_{fan_id}")
            lines.append(f"  - source: harness_{fan_id}")
            lines.append(f"    target: [dc_bus_group_{cluster}, {k}]")

        lines.append(f"  - source: [dc_bus_group_{cluster}, 1]")
        lines.append(f"    target: harness_central_{cluster}")
        lines.append(f"  - source: harness_central_{cluster}")
        lines.append(f"    target: dc_splitter_{cluster}")
        lines.append(f"  - source: [dc_splitter_{cluster}, 1]")
        lines.append(f"    target: dc_sspc_{cluster}1")
        lines.append(f"  - source: dc_sspc_{cluster}1")
        lines.append(f"    target: battery_pack_{cluster}")
        lines.append(f"  - source: [dc_splitter_{cluster}, 2]")
        lines.append(f"    target: dc_sspc_{cluster}2")
        lines.append(f"  - source: dc_sspc_{cluster}2")
        lines.append(f"    target: dc_dc_converter_{cluster}")
        lines.append(f"  - source: dc_dc_converter_{cluster}")
        lines.append(f"    target: pemfc_stack_{cluster}")
        lines.append(f"  - source: pemfc_stack_{cluster}")
        lines.append(f"    target: [h2_fuel_system_{cluster}, 1]")
        lines.append(f"  - source: [h2_fuel_system_{cluster}, 1]")
        lines.append(f"    target: gaseous_hydrogen_tank_{cluster}")

    lines.append(
        "watcher_file_path: ../results/ampere_final_power_train_data.csv"
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--fans-per-cluster", type=int, default=N_FANS_PER_CLUSTER)
    parser.add_argument(
        "--out",
        type=str,
        default="data/generated/assembly_ampere_final.yml",
    )
    args = parser.parse_args()

    content = generate_assembly_yaml(args.n_clusters, args.fans_per_cluster)
    with open(args.out, "w") as f:
        f.write(content)
    print(f"Wrote {args.out} ({args.n_clusters} clusters x {args.fans_per_cluster} fans = "
          f"{args.n_clusters * args.fans_per_cluster} total fans)")
