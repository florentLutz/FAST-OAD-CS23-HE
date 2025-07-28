# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from openmdao.vectors.default_transfer import DefaultTransfer


def filter_residuals(residuals: DefaultTransfer):
    # Function created to help screen for the residuals which contains nan

    filtered_residuals = {}

    for residual in residuals._views_flat:
        if (
            np.isnan(residuals._views_flat[residual]).any()
            or np.isinf(residuals._views_flat[residual]).any()
        ):
            filtered_residuals[residual] = residuals._views_flat[residual]

    return filtered_residuals
