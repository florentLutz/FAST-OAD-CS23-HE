"""Least Squared regression to obtain the constants for iron losses calculation."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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

    # Data extraction for each file
    frequency_column_05 = np.array(
        [
            155.76,
            268.69,
            377.73,
            486.76,
            595.79,
            712.62,
            813.86,
            938.47,
            1035.83,
            1156.54,
            1265.58,
            1382.40,
            1487.54,
            1596.57,
            1697.82,
            1826.32,
            1927.57,
            2048.29,
            2145.64,
            2258.57,
            2367.60,
            2484.42,
            2593.46,
            2714.17,
            2803.74,
            2924.45,
            3025.70,
            3154.21,
            3247.66,
            3372.27,
            3477.41,
            3594.24,
            3695.48,
            3820.09,
            3917.45,
            4026.48,
            4143.30,
            4256.23,
            4361.37,
            4482.09,
            4579.44,
            4688.47,
            4809.19,
            4914.33,
        ]
    )
    sp_iron_losses_column_05 = np.array(
        [
            11.04,
            11.04,
            11.04,
            11.04,
            17.83,
            17.83,
            17.83,
            17.83,
            17.83,
            17.83,
            17.83,
            28.02,
            28.62,
            28.62,
            34.81,
            34.81,
            34.81,
            38.21,
            48.40,
            55.19,
            58.59,
            65.38,
            72.17,
            75.57,
            82.36,
            82.36,
            85.76,
            102.74,
            102.74,
            109.53,
            116.33,
            123.12,
            126.52,
            126.52,
            136.71,
            136.71,
            146.89,
            157.08,
            157.08,
            160.48,
            163.88,
            174.07,
            177.46,
            184.26,
        ]
    )

    frequency_column_1 = np.array(
        [
            144.08,
            268.69,
            373.83,
            490.65,
            595.79,
            712.62,
            817.76,
            922.90,
            1035.83,
            1156.54,
            1253.89,
            1370.72,
            1471.96,
            1584.89,
            1686.14,
            1810.75,
            1915.89,
            2024.92,
            2122.27,
            2231.31,
            2340.34,
            2445.48,
            2554.52,
            2663.55,
            2760.90,
            2866.04,
            2978.97,
            3084.11,
            3185.36,
            3286.60,
            3399.53,
            3504.67,
            3613.71,
            3718.85,
            3823.99,
            3925.23,
            4022.59,
            4143.30,
            4240.65,
            4349.69,
            4454.83,
            4552.18,
            4661.22,
            4774.14,
            4871.50,
            4976.64,
        ]
    )
    sp_iron_losses_column_1 = np.array(
        [
            4.25,
            4.25,
            14.43,
            21.23,
            28.02,
            34.81,
            41.61,
            48.40,
            61.98,
            75.57,
            92.55,
            112.93,
            116.33,
            140.10,
            153.69,
            170.67,
            180.86,
            197.84,
            225.01,
            258.98,
            286.15,
            316.72,
            343.89,
            377.85,
            408.42,
            435.59,
            466.16,
            500.12,
            527.29,
            551.07,
            585.03,
            608.81,
            642.77,
            673.34,
            700.51,
            734.47,
            758.25,
            795.61,
            822.78,
            853.35,
            883.92,
            911.09,
            934.86,
            968.83,
            995.99,
            1026.56,
        ]
    )

    frequency_column_15 = np.array(
        [
            147.98,
            264.80,
            373.83,
            486.76,
            591.90,
            704.83,
            802.18,
            922.90,
            1024.14,
            1121.50,
            1222.74,
            1327.88,
            1429.13,
            1522.59,
            1619.94,
            1725.08,
            1830.22,
            1931.46,
            2024.92,
            2106.70,
            2180.69,
            2258.57,
            2340.34,
            2418.22,
            2500.00,
            2573.99,
            2651.87,
            2733.65,
            2811.53,
            2889.41,
            2967.29,
            3052.96,
            3123.05,
            3204.83,
            3278.82,
            3360.59,
            3438.47,
            3520.25,
            3602.02,
            3676.01,
            3761.68,
            3839.56,
            3917.45,
            3995.33,
            4069.31,
            4147.20,
            4225.08,
            4310.75,
            4384.74,
            4466.51,
            4540.50,
            4626.17,
            4700.16,
            4855.92,
            4941.59,
            4785.83,
        ]
    )
    sp_iron_losses_column_15 = np.array(
        [
            4.25,
            14.43,
            24.62,
            45.00,
            68.78,
            92.55,
            112.93,
            136.71,
            160.48,
            201.24,
            248.79,
            286.15,
            323.51,
            364.26,
            405.02,
            445.78,
            489.93,
            527.29,
            578.24,
            646.17,
            717.49,
            782.02,
            849.95,
            924.67,
            996.00,
            1060.53,
            1131.85,
            1199.78,
            1271.11,
            1342.43,
            1410.36,
            1481.68,
            1542.82,
            1617.54,
            1682.07,
            1756.79,
            1817.93,
            1892.65,
            1963.97,
            2028.51,
            2099.83,
            2167.76,
            2239.08,
            2310.41,
            2378.34,
            2442.87,
            2510.80,
            2585.52,
            2650.05,
            2721.37,
            2792.70,
            2867.42,
            2925.16,
            3071.20,
            3139.13,
            2999.88,
        ]
    )

    frequency_column_2 = np.array(
        [
            147.20,
            268.69,
            362.15,
            469.63,
            563.08,
            675.23,
            768.69,
            871.50,
            969.63,
            1049.07,
            1123.83,
            1207.94,
            1278.04,
            1352.80,
            1432.24,
            1507.01,
            1591.12,
            1665.89,
            1745.33,
            1820.09,
            1890.19,
            1974.30,
        ]
    )
    sp_iron_losses_column_2 = np.array(
        [
            13.76,
            30.06,
            54.51,
            91.19,
            131.95,
            176.78,
            221.62,
            270.52,
            311.28,
            364.26,
            437.63,
            515.07,
            580.28,
            653.64,
            722.93,
            796.29,
            865.57,
            943.01,
            1012.30,
            1077.51,
            1150.87,
            1224.24,
        ]
    )

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
        frequency_column_05,
        frequency_column_1,
        frequency_column_15,
        frequency_column_2,
    ]

    iron_losses_list = [
        sp_iron_losses_column_05,
        sp_iron_losses_column_1,
        sp_iron_losses_column_15,
        sp_iron_losses_column_2,
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
