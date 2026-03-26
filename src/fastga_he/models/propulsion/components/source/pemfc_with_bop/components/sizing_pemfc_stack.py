# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_pemfc_weight import SizingPEMFCStackBOPWeight
from .sizing_pemfc_power_density import SizingPEMFCStackBOPPowerDensity
from .sizing_pemfc_specific_power import SizingPEMFCStackBOPSpecificPower
from .sizing_pemfc_dimensions import SizingPEMFCStackBOPDimensions
from .sizing_pemfc_volume import SizingPEMFCStackBOPVolume
from .sizing_pemfc_cg_x import SizingPEMFCStackBOPCGX
from .sizing_pemfc_cg_y import SizingPEMFCStackBOPCGY
from .sizing_pemfc_drag import SizingPEMFCStackBOPDrag
from .subcomponents.sizing_pemfc_bop import SizingPEMFCBOP
from .cstr_pemfc_stack import ConstraintsPEMFCStack

from ..constants import POSSIBLE_POSITION


class SizingPEMFCStackBOP(om.Group):
    """Class that regroups all the subcomponents for PEMFC stack sizing computations."""

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the PEMFC stack, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare(
            name="model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
            "Aerostak 200W empirical polarization model is set as default.",
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

        # The followong option(s) is/are dummy option(s) to ensure compatibility
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the PEMFC stack is directly connected to a bus, a special mode is required to "
            "interface the two",
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air inlet",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        position = self.options["position"]
        model_fidelity = self.options["model_fidelity"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        compressor_connection = self.options["compressor_connection"]
        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_pemfc",
            subsys=ConstraintsPEMFCStack(
                pemfc_stack_bop_id=pemfc_stack_bop_id, model_fidelity=model_fidelity
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_specific_power",
            subsys=SizingPEMFCStackBOPSpecificPower(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_power_density",
            subsys=SizingPEMFCStackBOPPowerDensity(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )

        if compressor_connection:
            self.add_subsystem(
                name="pemfc_bop",
                subsys=SizingPEMFCBOP(
                    pemfc_stack_bop_id=pemfc_stack_bop_id,
                    model_fidelity=model_fidelity,
                    coolant_fluid_type=coolant_fluid_type,
                    compressor_id="compressor_1",
                    pipe_id="pipe_1",
                    air_inlet_id="air_inlet_1",
                    primary_heat_exchanger_id="primary_heat_exchanger_1",
                    secondary_heat_exchanger_id="secondary_heat_exchanger_1",
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
            name="pemfc_weight",
            subsys=SizingPEMFCStackBOPWeight(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_volume",
            subsys=SizingPEMFCStackBOPVolume(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_dimension",
            subsys=SizingPEMFCStackBOPDimensions(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_CG_x",
            subsys=SizingPEMFCStackBOPCGX(pemfc_stack_bop_id=pemfc_stack_bop_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_CG_y",
            subsys=SizingPEMFCStackBOPCGY(pemfc_stack_bop_id=pemfc_stack_bop_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "pemfc_drag_ls" if low_speed_aero else "pemfc_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPEMFCStackBOPDrag(
                    pemfc_stack_bop_id=pemfc_stack_bop_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
