# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
from ..constants import SUBMODEL_PERFORMANCES_PEMFC_MODELING_OPTION
import fastoad.api as oad
from .perf_pemfc_expect_power_density import (
    PerformancesPEMFCMaxPowerDensityFuelCellSystem,
    PerformancesPEMFCMaxPowerDensityFuelCellStack,
)
from .perf_pemfc_expect_specific_power import (
    PerformancesPEMFCMaxSpecificPowerFuelCellSystem,
    PerformancesPEMFCMaxSpecificPowerFuelCellStack,
)


oad.RegisterSubmodel.active_models[SUBMODEL_PERFORMANCES_PEMFC_MODELING_OPTION] = (
    "submodel.propulsion.performances.pemfc.modeling_option.system"
)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_MODELING_OPTION,
    "submodel.propulsion.performances.pemfc.modeling_option.system",
)
class PerformancesPEMFCModelingOptionSystem(om.Group):
    """
    This group provide the maximum power density and maximum specific power of the PEMFC.
    These two sizing indicators are derived from PEMFC systems includes all the BOPs except
    inlet compressors.
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
            name="pemfc_max_power_density",
            subsys=PerformancesPEMFCMaxPowerDensityFuelCellSystem(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_max_specific_power",
            subsys=PerformancesPEMFCMaxSpecificPowerFuelCellSystem(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_MODELING_OPTION,
    "submodel.propulsion.performances.pemfc.modeling_option.stack",
)
class PerformancesPEMFCModelingOptionStack(om.Group):
    """
    This group defines the PEMFC stack's maximum power density and maximum specific power.
    These sizing indicators are derived solely from the PEMFC itself, ensuring flexibility in
    configuring BOPs.
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
            name="pemfc_max_power_density",
            subsys=PerformancesPEMFCMaxPowerDensityFuelCellStack(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_max_specific_power",
            subsys=PerformancesPEMFCMaxSpecificPowerFuelCellStack(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )
