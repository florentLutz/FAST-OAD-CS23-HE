# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import atmosphere
from stdatm import AtmosphereWithPartials
from stdatm.state_parameters import AIR_GAS_CONSTANT

RHO_SL = AtmosphereWithPartials(0.0).density


class InitializeAirspeedDerivatives(om.ExplicitComponent):
    """Computes the d_vx_dt at each time step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.non_nul_pd = None

    def initialize(self):
        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):
        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        climb_idx = np.linspace(
            0,
            number_of_points_climb - 1,
            number_of_points_climb,
        ).astype(int)
        descent_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise,
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 1,
            number_of_points_descent,
        ).astype(int)
        self.non_nul_pd = np.concatenate((climb_idx, descent_idx))

        self.add_input(
            "true_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "equivalent_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "gamma", shape=number_of_points, val=np.full(number_of_points, np.nan), units="rad"
        )

        self.add_output(
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, 0.0), units="m/s**2"
        )

        # These are nil most of the time because sin(gamma) will be nil in cruise and reserve ...
        self.declare_partials(
            of="d_vx_dt",
            wrt=["altitude", "equivalent_airspeed", "true_airspeed"],
            method="exact",
            rows=self.non_nul_pd,
            cols=self.non_nul_pd,
        )
        # ... but the derivative wrt gamma won't be nil during cruise and reserve paradoxically
        self.declare_partials(
            of="d_vx_dt",
            wrt=["gamma"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"]

        atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)

        d_v_tas_dh = (
            -0.5
            * equivalent_airspeed
            * RHO_SL**0.5
            * atm.density**-1.5
            * atm.partial_density_altitude
        )
        d_vx_dt = d_v_tas_dh * true_airspeed * np.sin(gamma)

        outputs["d_vx_dt"] = d_vx_dt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"]

        atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)

        partials["d_vx_dt", "true_airspeed"] = (
            -0.5
            * equivalent_airspeed
            * RHO_SL**0.5
            * atm.density**-1.5
            * atm.partial_density_altitude
            * np.sin(gamma)
        )[self.non_nul_pd]
        partials["d_vx_dt", "gamma"] = (
            -0.5
            * equivalent_airspeed
            * RHO_SL**0.5
            * atm.density**-1.5
            * atm.partial_density_altitude
            * true_airspeed
            * np.cos(gamma)
        )
        partials["d_vx_dt", "equivalent_airspeed"] = (
            -0.5
            * RHO_SL**0.5
            * atm.density**-1.5
            * atm.partial_density_altitude
            * true_airspeed
            * np.sin(gamma)
        )[self.non_nul_pd]

        # Terms used for the computation of the pressure in stdatm
        coeff_b = 44330.78
        coeff_c = 5.25587611

        # This works only when we are below the tropopause
        hessian_pressure_altitude = (
            atmosphere
            * coeff_c
            * (coeff_c - 1.0)
            / coeff_b**2.0
            * (1 - (altitude / 44330.78)) ** 3.25587611
        )

        hessian_density_altitude = (
            (
                hessian_pressure_altitude * atm.temperature**3.0
                - 2.0
                * atm.partial_pressure_altitude
                * atm.partial_temperature_altitude
                * atm.temperature**2.0
                + 2.0 * atm.temperature * atm.pressure * atm.partial_temperature_altitude**2.0
            )
            / atm.temperature**4.0
            / AIR_GAS_CONSTANT
        )

        partials["d_vx_dt", "altitude"] = (
            (-0.5 * equivalent_airspeed * RHO_SL**0.5 * true_airspeed * np.sin(gamma))
            * (
                -1.5 * atm.density**-2.5 * atm.partial_density_altitude**2.0
                + atm.density**-1.5 * hessian_density_altitude
            )
        )[self.non_nul_pd]
