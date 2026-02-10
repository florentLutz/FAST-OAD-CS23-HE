# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesSwitchingLosses(om.ExplicitComponent):
    """Computation of switching losses for the IGBT and the diode."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("switching_frequency", units="Hz", val=np.full(number_of_points, np.nan))
        self.add_input("current_IGBT", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("current_diode", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:c",
            val=np.nan,
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:c",
            val=np.nan,
        )

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:c",
            val=np.nan,
        )

        self.add_input(
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_igbt_switching_losses",
            val=1.0,
            units="unitless",
            desc="K-factor to tune turn-on and turn-off switching losses in the IGBT module",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_diode_switching_losses",
            val=1.0,
            units="unitless",
            desc="K-factor to tune the reverse recovery switching losses in the diode",
        )

        self.add_output(
            "switching_losses_diode",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )
        self.add_output(
            "switching_losses_IGBT",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )

    def setup_partials(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="switching_losses_diode",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_rr:a",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_rr:b",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_rr:c",
                "settings:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":k_diode_switching_losses",
            ],
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="switching_losses_diode",
            wrt=["switching_frequency", "current_diode"],
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="switching_losses_IGBT",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_on:a",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_on:b",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_on:c",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_off:a",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_off:b",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":energy_off:c",
                "settings:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":k_igbt_switching_losses",
            ],
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="switching_losses_IGBT",
            wrt=["switching_frequency", "current_IGBT"],
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        a_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:a"
        ]
        b_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:b"
        ]
        c_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:c"
        ]

        a_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:a"
        ]
        b_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:b"
        ]
        c_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:c"
        ]

        a_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:a"
        ]
        b_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:b"
        ]
        c_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:c"
        ]

        f_sw = inputs["switching_frequency"]
        i_igbt = inputs["current_IGBT"]
        i_diode = inputs["current_diode"]

        k_losses_igbt = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_igbt_switching_losses"
        ]
        k_losses_diode = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_diode_switching_losses"
        ]

        loss_diode = f_sw * (a_rr / 2.0 + b_rr * i_diode / np.pi + c_rr * i_diode**2.0 / 4)
        loss_igbt = f_sw * (
            (a_on + a_off) / 2.0
            + (b_on + b_off) * i_igbt / np.pi
            + (c_on + c_off) * i_igbt**2.0 / 4
        )
        # a, b and c coefficient on reference were interpolated to give the results in J

        outputs["switching_losses_diode"] = loss_diode * k_losses_diode
        outputs["switching_losses_IGBT"] = loss_igbt * k_losses_igbt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        a_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:a"
        ]
        b_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:b"
        ]
        c_on = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:c"
        ]

        a_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:a"
        ]
        b_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:b"
        ]
        c_off = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_off:c"
        ]

        a_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:a"
        ]
        b_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:b"
        ]
        c_rr = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:c"
        ]

        f_sw = inputs["switching_frequency"]
        i_igbt = inputs["current_IGBT"]
        i_diode = inputs["current_diode"]

        k_losses_igbt = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_igbt_switching_losses"
        ]
        k_losses_diode = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_diode_switching_losses"
        ]

        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:a",
        ] = f_sw / 2.0 * k_losses_diode
        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:b",
        ] = f_sw * i_diode / np.pi * k_losses_diode
        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_rr:c",
        ] = f_sw * i_diode**2.0 / 4 * k_losses_diode
        partials["switching_losses_diode", "switching_frequency"] = (
            a_rr / 2.0 + b_rr * i_diode / np.pi + c_rr * i_diode**2.0 / 4
        ) * k_losses_diode
        partials["switching_losses_diode", "current_diode"] = (
            f_sw * (b_rr / np.pi + c_rr * i_diode / 2.0) * k_losses_diode
        )
        partials[
            "switching_losses_diode",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_diode_switching_losses",
        ] = f_sw * (a_rr / 2.0 + b_rr * i_diode / np.pi + c_rr * i_diode**2.0 / 4)

        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:a",
        ] = f_sw / 2.0 * k_losses_igbt
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:a",
        ] = f_sw / 2.0 * k_losses_igbt
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:b",
        ] = f_sw * i_igbt / np.pi * k_losses_igbt
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:b",
        ] = f_sw * i_igbt / np.pi * k_losses_igbt
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":energy_on:c",
        ] = f_sw * i_igbt**2.0 / 4 * k_losses_igbt
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":energy_off:c",
        ] = f_sw * i_igbt**2.0 / 4 * k_losses_igbt
        partials["switching_losses_IGBT", "switching_frequency"] = (
            (a_on + a_off) / 2.0
            + (b_on + b_off) * i_igbt / np.pi
            + (c_on + c_off) * i_igbt**2.0 / 4
        ) * k_losses_igbt
        partials["switching_losses_IGBT", "current_IGBT"] = (
            f_sw * ((b_on + b_off) / np.pi + (c_on + c_off) * i_igbt / 2.0) * k_losses_igbt
        )
        partials[
            "switching_losses_IGBT",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":k_igbt_switching_losses",
        ] = f_sw * (
            (a_on + a_off) / 2.0
            + (b_on + b_off) * i_igbt / np.pi
            + (c_on + c_off) * i_igbt**2.0 / 4
        )
