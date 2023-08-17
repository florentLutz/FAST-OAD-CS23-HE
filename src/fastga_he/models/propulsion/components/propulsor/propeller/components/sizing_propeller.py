# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_weight import SizingPropellerWeight
from .sizing_propeller_depth import SizingPropellerDepth
from .sizing_propeller_cg import SizingPropellerCGX
from .sizing_propeller_drag import SizingPropellerDrag
from ..components.sizing_propeller_ref_cl import SizingPropellerReferenceCl
from ..components.sizing_propeller_ref_chord import SizingPropellerReferenceChord
from ..components.sizing_propeller_radius_to_span_ratio import SizingPropellerDiameterToSpanRatio
from ..components.sizing_propeller_radius_to_chord_ratio import SizingPropellerDiameterToChordRatio
from ..components.sizing_propeller_flapped_span_ratio import SizingPropellerFlappedRatio
from ..components.sizing_propeller_wing_ac_distance import SizingPropellerWingACDistance
from ..components.sizing_propeller_wing_le_distance_ratio import SizingPropellerWingLEDistanceRatio

from .cstr_propeller import ConstraintsPropeller

from ..constants import POSSIBLE_POSITION


class SizingPropeller(om.Group):
    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the propeller, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_propeller",
            subsys=ConstraintsPropeller(propeller_id=propeller_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "propeller_wright",  # Like Orville and Wilbur
            SizingPropellerWeight(propeller_id=propeller_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "propeller_depth",
            SizingPropellerDepth(propeller_id=propeller_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "propeller_CG",
            SizingPropellerCGX(propeller_id=propeller_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "ref_cl",
            SizingPropellerReferenceCl(propeller_id=propeller_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            "ref_chord",
            SizingPropellerReferenceChord(propeller_id=propeller_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            "d_to_b_ratio",
            SizingPropellerDiameterToSpanRatio(propeller_id=propeller_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "d_to_c_ratio",
            SizingPropellerDiameterToChordRatio(propeller_id=propeller_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "flapped_ratio",
            SizingPropellerFlappedRatio(propeller_id=propeller_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "prop_to_ac_distance",
            SizingPropellerWingACDistance(propeller_id=propeller_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "prop_to_le_distance",
            SizingPropellerWingLEDistanceRatio(propeller_id=propeller_id, position=position),
            promotes=["data:*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "propeller_drag_ls" if low_speed_aero else "propeller_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPropellerDrag(
                    propeller_id=propeller_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
