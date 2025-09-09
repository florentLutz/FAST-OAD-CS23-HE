import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":

    def predict_losses(model, f, bm):
        sqrt_f = np.sqrt(f)
        sqrt_bm = np.sqrt(bm)

        # Build features as in training
        features = []
        for i_pow in range(1, 5):
            for j_pow in range(1, 5):
                features.append((sqrt_bm**j_pow) * (sqrt_f**i_pow))

        # Convert to 2D array for the model
        features = np.array(features).reshape(1, -1)

        # Prediction
        pf_pred = model.predict(features)[0]
        return pf_pred

    # Load CSV files
    df1 = pd.read_csv("bm0.5.csv", quotechar='"', delimiter=";")
    df2 = pd.read_csv("bm1.csv", quotechar='"', delimiter=";")
    df3 = pd.read_csv("bm1.5.csv", quotechar='"', delimiter=";")
    df4 = pd.read_csv("bm2.csv", quotechar='"', delimiter=";")

    # Remove spaces from column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()
    df4.columns = df4.columns.str.strip()

    # Data extraction for each file
    frequency_column_05 = df1["x"].to_numpy()
    sp_iron_losses_column_05 = df1["y"].to_numpy()

    frequency_column_1 = df2["x"].to_numpy()
    sp_iron_losses_column_1 = df2["y"].to_numpy()

    frequency_column_15 = df3["x"].to_numpy()
    sp_iron_losses_column_15 = df3["y"].to_numpy()

    frequency_column_2 = df4["x"].to_numpy()
    sp_iron_losses_column_2 = df4["y"].to_numpy()

    # Combine all data into a single matrix for regression
    frequencies = np.concatenate(
        [frequency_column_05, frequency_column_1, frequency_column_15, frequency_column_2]
    )
    iron_losses = np.concatenate(
        [
            sp_iron_losses_column_05,
            sp_iron_losses_column_1,
            sp_iron_losses_column_15,
            sp_iron_losses_column_2,
        ]
    )

    # Data for bm (all 4 values)
    bms = np.array([0.5, 1.0, 1.5, 2.0])

    # List of bm values
    bm_values = [0.5, 1.0, 1.5, 2.0]

    # Lists of arrays for frequencies and iron losses
    frequencies_list = [
        df1["x"].to_numpy(),
        df2["x"].to_numpy(),
        df3["x"].to_numpy(),
        df4["x"].to_numpy(),
    ]

    iron_losses_list = [
        df1["y"].to_numpy(),
        df2["y"].to_numpy(),
        df3["y"].to_numpy(),
        df4["y"].to_numpy(),
    ]

    # Prepare features combining necessary powers of sqrt(bm) and sqrt(f) up to degree 4
    X = []

    for i in range(len(bm_values)):
        f_i = frequencies_list[i]
        pf_i = iron_losses_list[i]
        bm_i = bm_values[i]
        for freq in f_i:
            sqrt_f = np.sqrt(freq)
            # Add combinations of (sqrt(bm))^j and (sqrt(f))^i for each bm
            features = []
            for i_pow in range(1, 5):  # i goes from 1 to 4
                for j_pow in range(1, 5):  # j goes from 1 to 4
                    features.append((np.sqrt(bm_i) ** j_pow) * (sqrt_f**i_pow))

            X.append(features)

    X = np.array(X)

    # Linear regression to find coefficients a_ij
    model = LinearRegression(fit_intercept=False)  # Without intercept, only coefficients a_ij
    model.fit(X, iron_losses)

    # Coefficients a_ij
    coeffs = model.coef_

    # Check the number of coefficients
    print(f"Number of coefficients: {len(coeffs)}")

    # Reshape coefficients into a 4x4 matrix (since i,j go from 1 to 4)
    coeffs_reshaped = coeffs.reshape(4, 4)
    print("Reshaped coefficient matrix (4x4):")
    print(coeffs_reshaped)

    # Colors and labels for each bm value
    bm_values = [0.5, 1.0, 1.5, 2.0]
    labels = [f"bₘ = {bm}" for bm in bm_values]
    colors = ["red", "green", "orange", "purple"]

    # Common frequency grid for plotting
    frequencies_plot = np.linspace(min(frequencies), max(frequencies), 100)

    # Plot original data
    plt.scatter(frequencies, iron_losses, label="Original data", color="blue", alpha=0.5)

    # Generate and plot curves for each bm
    for bm, label, color in zip(bm_values, labels, colors):
        sqrt_bm = np.sqrt(bm)
        X_plot = []
        for freq in frequencies_plot:
            sqrt_f = np.sqrt(freq)
            features_plot = []
            for i_pow in range(1, 5):
                for j_pow in range(1, 5):
                    features_plot.append((sqrt_bm**j_pow) * (sqrt_f**i_pow))
            X_plot.append(features_plot)

        X_plot = np.array(X_plot)
        predictions = model.predict(X_plot)

        plt.plot(frequencies_plot, predictions, label=label, color=color)

    # Plot details
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Iron losses (W/kg)")
    plt.title("Iron losses fitting for different values of bₘ")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example: prediction for f = 120 Hz and bm = 1.2 T
    f_test = 15970 / 30
    bm_test = 0.9

    pf_estimated = predict_losses(model, f_test, bm_test) * 224.88
    print(
        f"For f = {f_test} Hz and bm = {bm_test} T, the estimated losses are: {pf_estimated:.2f} W/kg"
    )
