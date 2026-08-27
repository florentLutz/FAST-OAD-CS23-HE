# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import scipy


class LCCAnnualDeliveryCount(om.Group):
    """
    Group that computes the annual delivery aircraft count.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_subsystem(
            name="production_fraction_factor",
            subsys=_ProductionFractionFactor(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="production_growth_factor",
            subsys=_ProductionGrowthFactor(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="annual_delivery_count",
            subsys=_LCCAnnualDeliveryCount(
                years_of_development=years_of_development,
                years_of_program=years_of_program,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="round_down",
            subsys=_RoundDown(
                years_of_development=years_of_development,
                years_of_program=years_of_program,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="cumulative_annual_delivery_count",
            subsys=_CumulativeAnnualDeliveryCount(
                years_of_development=years_of_development,
                years_of_program=years_of_program,
            ),
            promotes=["*"],
        )


class _ProductionFractionFactor(om.ExplicitComponent):
    """
    Computation of the production fraction factor of the aircraft. This factor is used to derive
    the production growth factor.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:cost:production:launch_aircraft_count",
            val=1,
            desc="The number of aircraft produced at the launch of deliveries",
        )
        self.add_input(
            "data:cost:production:annual_delivery_target",
            val=np.nan,
            desc="The planned target annual delivery count of aircraft.",
        )

        self.add_output(
            "production_fraction_factor",
            val=0.37,
            desc="The annual delivery count of aircraft over the program duration.",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        launch_aircraft_count = inputs["data:cost:production:launch_aircraft_count"]
        number_aircraft_5_years = inputs["data:cost:production:number_aircraft_5_years"]
        annual_delivery_target = inputs["data:cost:production:annual_delivery_target"]

        outputs["production_fraction_factor"] = 1.0 - (
            number_aircraft_5_years / 5.0 - launch_aircraft_count
        ) / (annual_delivery_target - launch_aircraft_count)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        launch_aircraft_count = inputs["data:cost:production:launch_aircraft_count"]
        number_aircraft_5_years = inputs["data:cost:production:number_aircraft_5_years"]
        annual_delivery_target = inputs["data:cost:production:annual_delivery_target"]

        common_denominator = annual_delivery_target - launch_aircraft_count

        partials["production_fraction_factor", "data:cost:production:number_aircraft_5_years"] = (
            -1.0 / (5.0 * common_denominator)
        )

        partials["production_fraction_factor", "data:cost:production:launch_aircraft_count"] = (
            5.0 * annual_delivery_target - number_aircraft_5_years
        ) / (5.0 * common_denominator**2.0)

        partials["production_fraction_factor", "data:cost:production:annual_delivery_target"] = (
            number_aircraft_5_years / 5.0 - launch_aircraft_count
        ) / common_denominator**2.0


class _ProductionGrowthFactor(om.ExplicitComponent):
    """
    Computation of the production growth factor of the aircraft. This factor is used to derive
    the annual delivery count of aircraft.
    """

    def setup(self):
        self.add_input(
            "production_fraction_factor",
            val=np.nan,
            desc="The production fraction factor of the aircraft.",
        )

        self.add_output(
            "production_growth_factor",
            val=0.4,
            desc="The production growth factor of the aircraft.",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        u = 1.0 / inputs["production_fraction_factor"]

        w = scipy.special.lambertw(-u * np.exp(-u), k=0).real

        outputs["production_growth_factor"] = 0.2 * (u + w)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        u = 1.0 / inputs["production_fraction_factor"]

        w = scipy.special.lambertw(-u * np.exp(-u), k=0).real
        f = 0.2 * (u + w)

        partials["production_growth_factor", "production_fraction_factor"] = -f / (
            inputs["production_fraction_factor"] * (1.0 + w)
        )


class _LCCAnnualDeliveryCount(om.ExplicitComponent):
    """
    Computation of the annual delivery aircraft count. The delivery count is based on the
    delivery history of Pilatus PC-12,
    https://pilatusowners.org/wp-content/uploads/2024/01/Pilatus-PC-12-Model-Variants-POPA-Members-Guide-Jan-2024.pdf.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_input(
            "production_growth_factor",
            val=np.nan,
            desc="The production growth factor of the aircraft.",
        )
        self.add_input(
            "data:cost:production:launch_aircraft_count",
            val=1,
            desc="The number of aircraft produced at the launch of deliveries",
        )
        self.add_input(
            "data:cost:production:annual_delivery_target",
            val=np.nan,
            desc="The planned target annual delivery count of aircraft.",
        )

        self.add_output(
            "original_annual_delivery_count",
            val=1.0,
            desc="The annual delivery count of aircraft over the program duration.",
            shape=years_of_program - years_of_development + 1,
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(years_of_program - years_of_development + 1),
            cols=np.zeros(years_of_program - years_of_development + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        production_growth_factor = inputs["production_growth_factor"]
        launch_aircraft_count = inputs["data:cost:production:launch_aircraft_count"]
        annual_delivery_target = inputs["data:cost:production:annual_delivery_target"]

        year_progression = np.linspace(
            0, years_of_program - years_of_development, years_of_program - years_of_development + 1
        )

        outputs["original_annual_delivery_count"] = launch_aircraft_count + (
            annual_delivery_target - launch_aircraft_count
        ) * (1.0 - np.exp(-production_growth_factor * year_progression))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        production_growth_factor = inputs["production_growth_factor"]
        launch_aircraft_count = inputs["data:cost:production:launch_aircraft_count"]
        annual_delivery_target = inputs["data:cost:production:annual_delivery_target"]

        year_progression = np.linspace(
            0, years_of_program - years_of_development, years_of_program - years_of_development + 1
        )

        partials[
            "original_annual_delivery_count",
            "data:cost:production:annual_delivery_target",
        ] = 1.0 - np.exp(-production_growth_factor * year_progression)

        partials[
            "original_annual_delivery_count",
            "data:cost:production:launch_aircraft_count",
        ] = 1.0 - (1.0 - np.exp(-production_growth_factor * year_progression))

        partials["original_annual_delivery_count", "production_growth_factor"] = (
            (annual_delivery_target - launch_aircraft_count)
            * year_progression
            * np.exp(-production_growth_factor * year_progression)
        )


class _RoundDown(om.ExplicitComponent):
    """
    Round down the annual delivery count to the nearest integer.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_input(
            "original_annual_delivery_count",
            val=np.nan,
            shape=years_of_program - years_of_development + 1,
        )

        self.add_output(
            "data:cost:production:annual_delivery_count",
            val=1,
            shape=years_of_program - years_of_development + 1,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:annual_delivery_count"] = np.floor(
            inputs["original_annual_delivery_count"]
        )


class _CumulativeAnnualDeliveryCount(om.ExplicitComponent):
    """
    Compute the cumulative annual delivery count of aircraft over the program duration.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_input(
            "data:cost:production:annual_delivery_count",
            val=np.nan,
            shape=years_of_program - years_of_development + 1,
            desc="The annual delivery count of aircraft over the program duration.",
        )

        self.add_output(
            "data:cost:production:cumulative_annual_delivery_count",
            val=np.nan,
            shape=years_of_program - years_of_development + 1,
            desc="The cumulative annual delivery count of aircraft over the program duration.",
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        rows, cols = np.tril_indices(years_of_program - years_of_development + 1)

        self.declare_partials(
            "data:cost:production:cumulative_annual_delivery_count",
            "data:cost:production:annual_delivery_count",
            rows=rows,
            cols=cols,
            val=np.ones(len(rows)),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:cumulative_annual_delivery_count"] = np.cumsum(
            inputs["data:cost:production:annual_delivery_count"]
        )
