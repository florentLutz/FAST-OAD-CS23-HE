#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from .pre_lca_capacity_loss import PreLCABatteryCapacityLoss
from .pre_lca_distance_to_target_loss import PreLCABatteryDistanceToTargetCapacityLoss

from ..constants import SERVICE_BATTERY_LIFESPAN

# I want the default behaviour to be that the lifespan is an input
oad.RegisterSubmodel.active_models[SERVICE_BATTERY_LIFESPAN] = None


@oad.RegisterSubmodel(
    SERVICE_BATTERY_LIFESPAN, "fastga_he.submodel.propulsion.battery.lifespan.legacy_aging_model"
)
class PreLCABatteryAging(om.Group):
    """
    Group that contain all the model pertaining to the battery aging model. Adaptation of the
    model from :cite:`chen:2019`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 10
        self.nonlinear_solver.options["rtol"] = 1e-3
        self.nonlinear_solver.options["atol"] = 1e-3
        self.linear_solver = om.DirectSolver()

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_subsystem(
            # Because of promotions shenanigans I'm too lazy to solve, I kinda need to do this.
            name=battery_pack_id + "_capacity_loss",
            subsys=PreLCABatteryCapacityLoss(battery_pack_id=battery_pack_id),
            promotes=[
                "data:*",
                (
                    "number_of_cycles",
                    "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
                ),
            ],
        )
        self.add_subsystem(
            name="distance_to_target",
            subsys=PreLCABatteryDistanceToTargetCapacityLoss(battery_pack_id=battery_pack_id),
            promotes=["data:*"],
        )

        self.connect(
            battery_pack_id + "_capacity_loss.capacity_loss_total",
            "distance_to_target.capacity_loss_total",
        )
