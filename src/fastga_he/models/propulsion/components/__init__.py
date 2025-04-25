# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

# pylint: disable=unused-import
# flake8: noqa

from .connectors.dc_bus import SizingDCBus, PerformancesDCBus, SlipstreamDCBus, PreLCADCBus
from .connectors.dc_cable import (
    SizingHarness,
    PerformancesHarness,
    SlipstreamHarness,
    PreLCAHarness,
    LCCHarnessCost,
)
from .connectors.dc_dc_converter import (
    SizingDCDCConverter,
    PerformancesDCDCConverter,
    SlipstreamDCDCConverter,
    PreLCADCDCConverter,
)
from .connectors.inverter import (
    SizingInverter,
    PerformancesInverter,
    SlipstreamInverter,
    PreLCAInverter,
)
from .connectors.dc_sspc import SizingDCSSPC, PerformancesDCSSPC, SlipstreamDCSSPC, PreLCADCSSPC
from .connectors.dc_splitter import (
    SizingDCSplitter,
    PerformancesDCSplitter,
    SlipstreamDCSplitter,
    PreLCADCSplitter,
)
from .connectors.rectifier import (
    SizingRectifier,
    PerformancesRectifier,
    SlipstreamRectifier,
    PreLCARectifier,
)
from .connectors.fuel_system import (
    SizingFuelSystem,
    PerformancesFuelSystem,
    SlipstreamFuelSystem,
    PreLCAFuelSystem,
)
from .connectors.speed_reducer import (
    SizingSpeedReducer,
    PerformancesSpeedReducer,
    SlipstreamSpeedReducer,
    PreLCASpeedReducer,
)
from .connectors.planetary_gear import (
    SizingPlanetaryGear,
    PerformancesPlanetaryGear,
    SlipstreamPlanetaryGear,
    PreLCAPlanetaryGear,
    LCCPlanetaryGearCost,
)
from .connectors.gearbox import (
    SizingGearbox,
    PerformancesGearbox,
    SlipstreamGearbox,
    PreLCAGearbox,
    LCCGearboxCost,
)
from .loads.pmsm import (
    SizingPMSM,
    PerformancesPMSM,
    SlipstreamPMSM,
    PreLCAPMSM,
    LCCPMSMCost,
    LCCPMSMOperation,
)
from .loads.dc_load import (
    SizingDCAuxLoad,
    PerformancesDCAuxLoad,
    SlipstreamDCAuxLoad,
    LCCDCLoadCost,
    LCCDCLoadOperation,
)
from .propulsor.propeller import (
    SizingPropeller,
    PerformancesPropeller,
    SlipstreamPropeller,
    PreLCAPropeller,
    LCCPropellerCost,
    LCCPropellerOperation,
)
from .source.battery import (
    SizingBatteryPack,
    PerformancesBatteryPack,
    SlipstreamBatteryPack,
    PreLCABatteryPack,
    LCCBatteryPackCost,
    LCCBatteryPackOperation,
)
from .source.generator import (
    SizingGenerator,
    PerformancesGenerator,
    SlipstreamGenerator,
    PreLCAGenerator,
    LCCGeneratorCost,
    LCCGeneratorOperation,
)
from .source.ice import (
    SizingICE,
    PerformancesICE,
    SlipstreamICE,
    PreLCAICE,
    LCCICECost,
    LCCICEOperation,
)
from .source.high_rpm_ice import (
    SizingHighRPMICE,
    PerformancesHighRPMICE,
    SlipstreamHighRPMICE,
    PreLCAHighRPMICE,
    LCCHighRPMICECost,
    LCCHighRPMICEOperation,
)
from .source.turboshaft import (
    SizingTurboshaft,
    PerformancesTurboshaft,
    SlipstreamTurboshaft,
    PreLCATurboshaft,
    LCCTurboshaftCost,
    LCCTurboshaftOperation,
)
from .source.simple_turbo_generator import (
    SizingTurboGenerator,
    PerformancesTurboGenerator,
    SlipstreamTurboGenerator,
    PreLCATurboGenerator,
    LCCTurboGeneratorCost,
    LCCTurboGeneratorOperation,
)
from .source.pemfc import (
    SizingPEMFCStack,
    PerformancesPEMFCStack,
    SlipstreamPEMFCStack,
    LCCPEMFCStackCost,
    LCCPEMFCStackOperation,
)
from .tanks.fuel_tanks import (
    SizingFuelTank,
    PerformancesFuelTank,
    SlipstreamFuelTank,
    PreLCAFuelTank,
    LCCFuelTankCost,
    LCCFuelTankOperation,
)
from .tanks.gaseous_hydrogen_tank import (
    SizingGaseousHydrogenTank,
    PerformancesGaseousHydrogenTank,
    SlipstreamGaseousHydrogenTank,
    LCCGaseousHydrogenTankCost,
    LCCGaseousHydrogenTankOperation,
)
from .connectors.h2_fuel_system import (
    SizingH2FuelSystem,
    PerformancesH2FuelSystem,
    SlipstreamH2FuelSystem,
)
