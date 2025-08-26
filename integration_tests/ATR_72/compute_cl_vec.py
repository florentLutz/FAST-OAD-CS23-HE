import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Wing span
    c_root = 2.655
    c_tip = 1.70
    b = 27.05  # in meters
    S_w2 = c_tip * c_root * b / 2
    S_w = 61  # in m**2

    # Create a vector of spanwise positions (y) from -b/2 to b/2 (for a symmetric wing)
    y = np.linspace(0, b / 2, 20)

    # Calculate the elliptic lift distribution at each spanwise location y
    L_y = np.sqrt(1 - (y / (b / 2)) ** 2)

    ch_vec = np.linspace(c_root, c_tip, 20)

    # Multiply L_y by the chord vector (ch_vec)
    product = L_y * ch_vec

    # Perform the numerical integration using the trapezoidal rule
    integral_result = np.trapz(product, y)

    CL_ref = integral_result / (S_w / 2)
    print(f"CL_ref: {CL_ref} ")
    Cl0_ref = 0.3
    CL_vec = L_y * 0.3 / 0.7832967521264312
    product2 = CL_vec * ch_vec
    integral_result2 = np.trapz(product2, y)

    CL_ref2 = integral_result2 / (S_w / 2)
    print(f"CL_ref: {CL_ref2} ")

    print("CHORD VEC:", ch_vec)
    print("Y POS :", y)
    print("Cl_vec:", CL_vec)

    # Plot the lift distribution
    plt.plot(y, ch_vec)
    plt.title("Elliptic Lift Distribution")
    plt.xlabel("Spanwise Location (y)")
    plt.ylabel("Lift at y")
    plt.grid(True)
    plt.show()
