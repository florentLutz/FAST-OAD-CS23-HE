# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from fastoad.io import VariableIO
from fastoad.openmdao.variables import VariableList


def write_outputs(file_path: str, problem: om.Problem):
    writer = VariableIO(file_path)
    variables = VariableList.from_problem(problem, promoted_only=True)
    writer.write(variables)
