# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ...components.source.pemfc_with_bop.components.subcomponents.sizing_pemfc_bop import (
    SizingPEMFCBOP,
)
from ...components.source.pemfc_with_bop.components.subcomponents.perf_pemfc_bop import (
    PerformancesPEMFCBOP,
)


class FullSimpleBOPAssembly(om.Group):
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

        self.add_subsystem(
            name="sizing",
            subsys=SizingPEMFCBOP(
                pemfc_stack_bop_id="pemfc_stack_bop_1",
                coolant_fluid_type="ethylene glycol",
                compressor_id="compressor_1",
                pipe_id="pipe_1",
                air_inlet_id="air_inlet_1",
                primary_heat_exchanger_id="primary_heat_exchanger_1",
                supplement_heat_exchanger_id="supplement_heat_exchanger_1",
                humidifier_id="humidifier_1",
                diffuser_id="diffuser_1",
                nozzle_id="nozzle_1",
                pump_id="pump_1",
                valve_id="valve_1",
                coolant_tank_id="coolant_tank_1",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="performances",
            subsys=PerformancesPEMFCBOP(
                number_of_points=number_of_points,
                pemfc_stack_bop_id="pemfc_stack_bop_1",
                coolant_fluid_type="ethylene glycol",
                compressor_id="compressor_1",
                pipe_id="pipe_1",
                air_inlet_id="air_inlet_1",
                primary_heat_exchanger_id="primary_heat_exchanger_1",
                supplement_heat_exchanger_id="supplement_heat_exchanger_1",
                humidifier_id="humidifier_1",
                diffuser_id="diffuser_1",
                nozzle_id="nozzle_1",
                pump_id="pump_1",
            ),
            promotes=["*"],
        )
