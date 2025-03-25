# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_ice_rotax.xml"
NB_POINTS_TEST = 10
