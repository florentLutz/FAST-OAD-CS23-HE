# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
from ..constants import SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION
import fastoad.api as oad
from .sizing_pemfc_weight import SizingPEMFCWeightAerostak200W, SizingPEMFCWeightSpecificPower
from .sizing_pemfc_dimensions import (
    SizingPEMFCDimensionsAerostak200W,
    SizingPEMFCDimensionsPowerDensity,
)

oad.RegisterSubmodel.active_models[SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION] = (
    "fastga_he.submodel.propulsion.sizing.pemfc.weight_dimension.aerostak200"
)


@oad.RegisterSubmodel(
    SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION,
    "fastga_he.submodel.propulsion.sizing.pemfc.weight_dimension.aerostak200",
)
class SizingPEMFCWeightDimensionsAerostak200W(om.Group):
    """
    Collect both weight and dimension of PEMFC based on Aerostak 200W.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_subsystem(
            name="pemfc_weight",
            subsys=SizingPEMFCWeightAerostak200W(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_dimension",
            subsys=SizingPEMFCDimensionsAerostak200W(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )


@oad.RegisterSubmodel(
    SUBMODEL_SIZING_PEMFC_WEIGHT_DIMENSION,
    "fastga_he.submodel.propulsion.sizing.pemfc.weight_dimension.adjusted",
)
class SizingPEMFCWeightDimensionsAdjusted(om.Group):
    """
    Collect both weight and dimension of PEMFC using power density and specific power.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_subsystem(
            name="pemfc_weight",
            subsys=SizingPEMFCWeightSpecificPower(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_dimension",
            subsys=SizingPEMFCDimensionsPowerDensity(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
