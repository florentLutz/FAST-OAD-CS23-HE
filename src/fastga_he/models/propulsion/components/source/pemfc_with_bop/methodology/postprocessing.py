import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Data
    configurations = [
        "Original Configuration",
        "PEMFC + PT6A-112",
        "PEMFC + PW206b",
        "Pure PEMFC-GH2",
    ]
    owe = [3475.4, 3802.1, 4039.7, 5128.8]
    design_payload = [1375, 1063.1, 1030.5, 406.7]
    power_train_mass = [595.3, 882.4, 1115.1, 2177.9]
    airframe_mass = [1838.3, 1877.3, 1882.1, 1908.6]
    kerosene_mass = [789.1, 774.0, 554.6, 0]
    hydrogen_mass = [0.0, 0.3, 14.7, 104.0]

    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Set width of bars
    bar_width = 0.2

    # Set positions of bars on x-axis
    r1 = np.arange(len(configurations))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Create bars
    bars1 = ax1.bar(r1, owe, color="#8B9DC9", width=bar_width, label="OWE")
    bars2 = ax1.bar(r2, design_payload, color="#6B78B4", width=bar_width, label="Design Payload")
    bars3 = ax1.bar(
        r3, power_train_mass, color="#5D4E95", width=bar_width, label="Power Train Mass"
    )
    bars4 = ax1.bar(r4, airframe_mass, color="#453A7D", width=bar_width, label="Airframe Mass")

    # Create stacked bars for fuel mass
    bars5_kerosene = ax1.bar(r5, kerosene_mass, color="#4A3B7F", width=bar_width, label="Kerosene")
    bars5_hydrogen = ax1.bar(
        r5, hydrogen_mass, bottom=kerosene_mass, color="#6A5B9F", width=bar_width, label="Hydrogen"
    )

    # Function to add value labels
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Add value labels
    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)
    add_value_labels(ax1, bars3)
    add_value_labels(ax1, bars4)

    # Add value labels for stacked fuel mass
    for i, (k, h) in enumerate(zip(kerosene_mass, hydrogen_mass)):
        total = k + h
        ax1.text(r5[i], total, f"{total:.1f}", ha="center", va="bottom", fontsize=10)
        if k > 0:
            ax1.text(r5[i], k / 2, f"{k:.1f}", ha="center", va="center", fontsize=10, color="white")
        if h > 0:
            ax1.text(
                r5[i], k + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=10, color="white"
            )

    # Add labels and title
    ax1.set_xlabel("Configuration", fontsize=14)
    ax1.set_ylabel("Weight (kg)", fontsize=14)
    plt.title(
        "Aerostak PEMFC-GH2 turboshaft hybrid powered retrofit aircraft Comparison", fontsize=20
    )

    # Adjust x-axis
    plt.xticks(
        [r + 1.5 * bar_width for r in range(len(configurations))],
        configurations,
        ha="center",
        fontsize=12,
        rotation=15,
    )

    # Add legends
    ax1.legend(loc="upper right", fontsize=12)

    # Adjust layout and display the chart
    plt.tight_layout()
    plt.show()
