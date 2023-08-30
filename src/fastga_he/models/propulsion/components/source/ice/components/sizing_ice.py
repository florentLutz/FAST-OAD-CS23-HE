# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_displacement_volume import SizingICEDisplacementVolume
from ..components.sizing_ice_uninstalled_weight import SizingICEUninstalledWeight
from ..components.sizing_ice_weight import SizingICEWeight
from ..components.sizing_ice_dimensions_scaling import SizingICEDimensionsScaling
from ..components.sizing_ice_dimensions import SizingICEDimensions
from ..components.sizing_ice_nacelle_dimensions import SizingICENacelleDimensions
from ..components.sizing_ice_nacelle_wet_area import SizingICENacelleWetArea
from ..components.sizing_ice_cg_x import SizingICECGX
from ..components.sizing_ice_cg_y import SizingICECGY
from ..components.sizing_ice_drag import SizingICEDrag

from ..components.cstr_ice import ConstraintsICE

from ..constants import POSSIBLE_POSITION


class SizingICE(om.Group):
    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_ice",
            subsys=ConstraintsICE(ice_id=ice_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="displacement_volume",
            subsys=SizingICEDisplacementVolume(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="uninstalled_weight",
            subsys=SizingICEUninstalledWeight(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="installed_weight",
            subsys=SizingICEWeight(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dimensions_scaling",
            subsys=SizingICEDimensionsScaling(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_dimensions",
            subsys=SizingICEDimensions(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_dimensions",
            subsys=SizingICENacelleDimensions(ice_id=ice_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="nacelle_wet_area",
            subsys=SizingICENacelleWetArea(ice_id=ice_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_cg_x",
            subsys=SizingICECGX(ice_id=ice_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="ice_cg_y",
            subsys=SizingICECGY(ice_id=ice_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "ice_drag_ls" if low_speed_aero else "ice_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingICEDrag(
                    ice_id=ice_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
