# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_simple_assembly_gh2_pemfc import SizingAssembly
from .performances_simple_assembly_gh2_pemfc import PerformancesAssembly
from ...assemblers.sizing_from_pt_file import PowerTrainSizingFromFile
from ...assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile


class FullSimpleAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(name="sizing", subsys=SizingAssembly(), promotes=["*"])
        self.add_subsystem(
            name="performances",
            subsys=PerformancesAssembly(number_of_points=number_of_points),
            promotes=["*"],
        )


class FullSimpleAssemblyPT(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="add_solver",
            default=True,
            desc="Boolean to add solvers to the power train performance group. Default is False "
            "it can be turned off when used jointly with the mission to save computation time",
            allow_none=False,
        )
        self.options.declare(
            name="pre_condition_pt",
            default=False,
            desc="Boolean to pre_condition the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        power_train_file_path = self.options["power_train_file_path"]
        add_solver = self.options["add_solver"]
        pre_condition_pt = self.options["pre_condition_pt"]

        self.add_subsystem(
            name="sizing",
            subsys=PowerTrainSizingFromFile(power_train_file_path=power_train_file_path),
            promotes=["*"],
        )
        self.add_subsystem(
            name="performances",
            subsys=PowerTrainPerformancesFromFile(
                number_of_points=number_of_points,
                power_train_file_path=power_train_file_path,
                add_solver=add_solver,
                pre_condition_pt=pre_condition_pt,
            ),
            promotes=["*"],
        )
