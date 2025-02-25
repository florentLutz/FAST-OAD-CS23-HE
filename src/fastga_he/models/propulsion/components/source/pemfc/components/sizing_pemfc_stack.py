# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_pemfc_weight import SizingPEMFCStackWeight
from .sizing_pemfc_power_density import SizingPEMFCStackPowerDensity
from .sizing_pemfc_specific_power import SizingPEMFCStackSpecificPower
from .sizing_pemfc_dimensions import SizingPEMFCStackDimensions
from .sizing_pemfc_volume import SizingPEMFCStackVolume
from .sizing_pemfc_cg_x import SizingPEMFCStackCGX
from .sizing_pemfc_cg_y import SizingPEMFCStackCGY
from .sizing_pemfc_drag import SizingPEMFCStackDrag
from .cstr_pemfc_stack import ConstraintsPEMFCStack

from ..constants import POSSIBLE_POSITION


class SizingPEMFCStack(om.Group):
    """Class that regroups all the subcomponents for PEMFC stack sizing computation."""

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of PEMFC, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare(
            "model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
                 "Aerostak 200W empirical polarization model is set as default.",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        model_fidelity = self.options["model_fidelity"]
        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_pemfc",
            subsys=ConstraintsPEMFCStack(
                pemfc_stack_id=pemfc_stack_id, model_fidelity=model_fidelity
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_specific_power",
            subsys=SizingPEMFCStackSpecificPower(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_power_density",
            subsys=SizingPEMFCStackPowerDensity(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_weight",
            subsys=SizingPEMFCStackWeight(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_volume",
            subsys=SizingPEMFCStackVolume(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_dimension",
            subsys=SizingPEMFCStackDimensions(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_CG_x",
            subsys=SizingPEMFCStackCGX(pemfc_stack_id=pemfc_stack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_CG_y",
            subsys=SizingPEMFCStackCGY(pemfc_stack_id=pemfc_stack_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "pemfc_drag_ls" if low_speed_aero else "pemfc_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPEMFCStackDrag(
                    pemfc_stack_id=pemfc_stack_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
