# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
"""
This module is to study the effect of temperature on battery internal resistance. One of the
reference in :cite:`vratny:2013` suggests that internal resistance is a product of a function of
SOC and a function of temperature uneder the form of an exponential. :cite:`lai:2019` gives the
evolution of internal resistance for the temperature for a 18650 cell (NCR 18650-type cylindrical
Li-ion). We will make the assumption that the coefficient in the exponential do not vary too much
from one Li-ion cell to another (Linked to chemistry, catalytic speed ?).
"""

import os.path as pth

import numpy as np
import pandas as pd

import scipy.optimize as opt
import plotly.graph_objects as go

if __name__ == "__main__":

    data_file = pth.join(pth.dirname(__file__), "data/internal_resistance_temperature.csv")
