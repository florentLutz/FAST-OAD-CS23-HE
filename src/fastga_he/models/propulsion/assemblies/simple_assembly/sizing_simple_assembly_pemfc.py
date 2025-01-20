# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ...components.loads.pmsm import SizingPMSM
from ...components.propulsor.propeller import SizingPropeller
from ...components.connectors.inverter import SizingInverter
from ...components.connectors.dc_cable import SizingHarness
from ...components.connectors.dc_bus import SizingDCBus
from ...components.connectors.dc_dc_converter import SizingDCDCConverter
from ...components.connectors.dc_sspc import SizingDCSSPC
from ...components.source.pemfc import SizingPEMFCStack
from ...components.tanks.hydrogen_gas_tank import SizingHydrogenGasTank


class SizingAssembly(om.Group):
    def setup(self):

        self.add_subsystem(
            "propeller_1",
            SizingPropeller(propeller_id="propeller_1"),
            promotes=["*"],
        )
        self.add_subsystem(
            "motor_1",
            SizingPMSM(motor_id="motor_1"),
            promotes=["*"],
        )
        self.add_subsystem(
            "inverter_1",
            SizingInverter(inverter_id="inverter_1"),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_bus_1",
            SizingDCBus(
                dc_bus_id="dc_bus_1",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_line_1",
            SizingHarness(
                harness_id="harness_1",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_bus_2",
            SizingDCBus(
                dc_bus_id="dc_bus_2",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_dc_converter_1",
            SizingDCDCConverter(
                dc_dc_converter_id="dc_dc_converter_1",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "pemfc_stack_1",
            SizingPEMFCStack(
                pemfc_stack_id="pemfc_stack_1",
                position="wing_pod",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "hydrogen_gas_tank_1",
            SizingHydrogenGasTank(
                hydrogen_gas_tank_id="hydrogen_gas_tank_1",
                position="wing_pod",
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "dc_sspc_1",
            SizingDCSSPC(dc_sspc_id="dc_sspc_1"),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_sspc_2",
            SizingDCSSPC(dc_sspc_id="dc_sspc_2"),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_sspc_412",
            SizingDCSSPC(
                dc_sspc_id="dc_sspc_412",
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_sspc_1337",
            SizingDCSSPC(dc_sspc_id="dc_sspc_1337"),
            promotes=["*"],
        )
