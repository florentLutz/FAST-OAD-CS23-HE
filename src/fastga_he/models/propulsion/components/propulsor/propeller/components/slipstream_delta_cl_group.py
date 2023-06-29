# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.slipstream_thrust_loading import SlipstreamPropellerThrustLoading
from ..components.slipstream_axial_induction_factor import SlipstreamPropellerAxialInductionFactor
from ..components.slipstream_contraction_ratio_squared import (
    SlipstreamPropellerContractionRatioSquared,
)
from ..components.slipstream_contraction_ratio import SlipstreamPropellerContractionRatio
from ..components.slipstream_axial_induction_factor_ac import (
    SlipstreamPropellerAxialInductionFactorWingAC,
)
from ..components.slipstream_axial_induction_factor_downstream import (
    SlipstreamPropellerVelocityRatioDownstream,
)
from ..components.slipstream_height_impact_coefficients import (
    SlipstreamPropellerHeightImpactCoefficients,
)
from ..components.slipstream_height_impact import SlipstreamPropellerHeightImpact
from ..components.slipstream_lift_increase_ratio import SlipstreamPropellerLiftIncreaseRatio
from ..components.slipstream_section_lift import SlipstreamPropellerSectionLift
from ..components.slipstream_delta_cl_2d import SlipstreamPropellerDeltaCl2D
from ..components.slipstream_blown_area_ratio import SlipstreamPropellerBlownAreaRatio
from ..components.slipstream_delta_cl import SlipstreamPropellerDeltaCl


class SlipstreamPropellerDeltaClGroup(om.Group):
    """
    Grouping the components that computes the delta Cl caused by the propeller because the
    computation is required twice. Once for the actual flight AoA one for AoA = 0.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
            values=["cruise", "takeoff", "landing"],
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        self.add_subsystem(
            name="thrust_loading",
            subsys=SlipstreamPropellerThrustLoading(
                propeller_id=propeller_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="axial_induction_factor",
            subsys=SlipstreamPropellerAxialInductionFactor(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="contraction_ratio_squared",
            subsys=SlipstreamPropellerContractionRatioSquared(
                propeller_id=propeller_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="contraction_ratio",
            subsys=SlipstreamPropellerContractionRatio(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="axial_induction_factor_wing_ac",
            subsys=SlipstreamPropellerAxialInductionFactorWingAC(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="velocity_ratio_downstream",
            subsys=SlipstreamPropellerVelocityRatioDownstream(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="height_impact_coefficient",
            subsys=SlipstreamPropellerHeightImpactCoefficients(
                propeller_id="propeller_1", number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="height_impact",
            subsys=SlipstreamPropellerHeightImpact(
                propeller_id="propeller_1", number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="lift_increase_ratio",
            subsys=SlipstreamPropellerLiftIncreaseRatio(
                propeller_id="propeller_1", number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="section_lift",
            subsys=SlipstreamPropellerSectionLift(
                propeller_id="propeller_1",
                number_of_points=number_of_points,
                flaps_position=flaps_position,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cl_2D",
            subsys=SlipstreamPropellerDeltaCl2D(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="blown_area_ratio",
            subsys=SlipstreamPropellerBlownAreaRatio(
                propeller_id="propeller_1", number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cl",
            subsys=SlipstreamPropellerDeltaCl(number_of_points=number_of_points),
            promotes=["*"],
        )
