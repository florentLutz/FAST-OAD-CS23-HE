# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_displacement_volume import SizingHighRPMICEDisplacementVolume
from ..components.sizing_high_rpm_ice_uninstalled_weight import SizingHighRPMICEUninstalledWeight
from ..components.sizing_high_rpm_ice_weight import SizingHighRPMICEWeight
from ..components.sizing_high_rpm_ice_dimensions_scaling import SizingHighRPMICEDimensionsScaling
from ..components.sizing_high_rpm_ice_dimensions import SizingHighRPMICEDimensions
from ..components.sizing_high_rpm_ice_nacelle_dimensions import SizingHighRPMICENacelleDimensions
from ..components.sizing_high_rpm_ice_nacelle_wet_area import SizingHighRPMICENacelleWetArea
from ..components.sizing_high_rpm_ice_cg_x import SizingHighRPMICECGX
from ..components.sizing_high_rpm_ice_cg_y import SizingHighRPMICECGY
from ..components.sizing_high_rpm_ice_drag import SizingHighRPMICEDrag

from ..components.cstr_high_rpm_ice import ConstraintsHighRPMICE

from ..constants import POSSIBLE_POSITION


class SizingHighRPMICE(om.Group):
    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the ICE, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_ice",
            subsys=ConstraintsHighRPMICE(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="displacement_volume",
            subsys=SizingHighRPMICEDisplacementVolume(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="uninstalled_weight",
            subsys=SizingHighRPMICEUninstalledWeight(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="installed_weight",
            subsys=SizingHighRPMICEWeight(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dimensions_scaling",
            subsys=SizingHighRPMICEDimensionsScaling(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_dimensions",
            subsys=SizingHighRPMICEDimensions(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_dimensions",
            subsys=SizingHighRPMICENacelleDimensions(
                high_rpm_ice_id=high_rpm_ice_id, position=position
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_wet_area",
            subsys=SizingHighRPMICENacelleWetArea(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_cg_x",
            subsys=SizingHighRPMICECGX(high_rpm_ice_id=high_rpm_ice_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_cg_y",
            subsys=SizingHighRPMICECGY(high_rpm_ice_id=high_rpm_ice_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "ice_drag_ls" if low_speed_aero else "ice_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingHighRPMICEDrag(
                    high_rpm_ice_id=high_rpm_ice_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
