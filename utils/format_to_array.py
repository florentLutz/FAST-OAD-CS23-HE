# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np


def format_to_array(input_array: np.ndarray, number_of_points: int) -> np.ndarray:
    """
    Takes an inputs which is either a one-element array or a multi-element array and formats it.
    """

    if len(input_array):
        output_array = np.full(number_of_points, input_array[0])
    else:
        output_array = input_array

    return output_array
