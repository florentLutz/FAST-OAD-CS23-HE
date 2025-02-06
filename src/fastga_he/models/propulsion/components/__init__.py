# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# pylint: disable=unused-import
# flake8: noqa

from .connectors.dc_bus import SizingDCBus, PerformancesDCBus, SlipstreamDCBus, PreLCADCBus
from .connectors.dc_cable import (
    SizingHarness,
    PerformancesHarness,
    SlipstreamHarness,
    PreLCAHarness,
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
)
from .connectors.gearbox import SizingGearbox, PerformancesGearbox, SlipstreamGearbox, PreLCAGearbox
from .loads.pmsm import SizingPMSM, PerformancesPMSM, SlipstreamPMSM, PreLCAPMSM
from .loads.dc_load import SizingDCAuxLoad, PerformancesDCAuxLoad, SlipstreamDCAuxLoad
from .propulsor.propeller import (
    SizingPropeller,
    PerformancesPropeller,
    SlipstreamPropeller,
    PreLCAPropeller,
)
from .source.battery import (
    SizingBatteryPack,
    PerformancesBatteryPack,
    SlipstreamBatteryPack,
    PreLCABatteryPack,
)
from .source.generator import (
    SizingGenerator,
    PerformancesGenerator,
    SlipstreamGenerator,
    PreLCAGenerator,
)
from .source.ice import SizingICE, PerformancesICE, SlipstreamICE, PreLCAICE
from .source.turboshaft import (
    SizingTurboshaft,
    PerformancesTurboshaft,
    SlipstreamTurboshaft,
    PreLCATurboshaft,
)
from .source.simple_turbo_generator import (
    SizingTurboGenerator,
    PerformancesTurboGenerator,
    SlipstreamTurboGenerator,
    PreLCATurboGenerator,
)

from .tanks.fuel_tanks import (
    SizingFuelTank,
    PerformancesFuelTank,
    SlipstreamFuelTank,
    PreLCAFuelTank,
)
from .tanks.gaseous_hydrogen_tank import (
    SizingGaseousHydrogenTank,
    PerformancesGaseousHydrogenTank,
    SlipstreamGaseousHydrogenTank,
)
