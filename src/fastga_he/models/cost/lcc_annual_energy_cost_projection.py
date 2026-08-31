# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalAnnualEnergyCostProjection(om.Group):
    """
    Group that computes the sum of the annual energy cost projection for each energy source over
    the service life of the aircraft.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_subsystem(
            "compound_annual_gain_factor",
            _CompoundAnnualGainFactor(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            "annual_energy_cost_projection",
            _AnnualEnergyCostProjection(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            "energy_cost_projection_sum",
            _EnergyCostProjectionSum(years_of_service=years_of_service),
            promotes=["*"],
        )


class _CompoundAnnualGainFactor(om.ExplicitComponent):
    """
    Computation of the compound annual gain factor based on the compound annual growth rate (CAGR)
    for each energy source over the service life of the aircraft. The hydrocarbon fuel CAGR is
    derived the CAGR of petroleum prices history from
    https://www.macrotrends.net/1369/crude-oil-price-history-chart. The hydrogen fuel CAGR is
    derived from the
    CAGR of hydrogen prices history from
    https://businessanalytiq.com/procurementanalytics/index/hydrogen-price-index/. The electricity
    CAGR is derived from the CAGR of electricity prices history from
    https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            name="data:cost:operation:hydrocarbon_fuel_compound_annual_gain_rate",
            val=0.00543,
        )
        self.add_input(
            name="data:cost:operation:hydrogen_fuel_compound_annual_gain_rate",
            val=0.033,
        )
        self.add_input(
            name="data:cost:operation:electricity_compound_annual_gain_rate",
            val=0.032,
        )

        self.add_output(
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
            val=1.0,
            shape=years_of_service,
        )
        self.add_output(
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
            val=1.0,
            shape=years_of_service,
        )
        self.add_output(
            "data:cost:operation:electricity_compound_annual_gain_factor",
            val=1.0,
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_rate",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

        self.declare_partials(
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_rate",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

        self.declare_partials(
            "data:cost:operation:electricity_compound_annual_gain_factor",
            "data:cost:operation:electricity_compound_annual_gain_rate",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_service = self.options["years_of_service"]

        hydrocarbon_fuel_cagr = inputs[
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_rate"
        ]
        hydrogen_fuel_cagr = inputs["data:cost:operation:hydrogen_fuel_compound_annual_gain_rate"]
        electricity_cagr = inputs["data:cost:operation:electricity_compound_annual_gain_rate"]

        outputs["data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor"] = (
            1.0 + hydrocarbon_fuel_cagr
        ) ** np.arange(years_of_service)
        outputs["data:cost:operation:hydrogen_fuel_compound_annual_gain_factor"] = (
            1.0 + hydrogen_fuel_cagr
        ) ** np.arange(years_of_service)
        outputs["data:cost:operation:electricity_compound_annual_gain_factor"] = (
            1.0 + electricity_cagr
        ) ** np.arange(years_of_service)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        hydrocarbon_fuel_cagr = inputs[
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_rate"
        ]
        hydrogen_fuel_cagr = inputs["data:cost:operation:hydrogen_fuel_compound_annual_gain_rate"]
        electricity_cagr = inputs["data:cost:operation:electricity_compound_annual_gain_rate"]

        partials[
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_rate",
        ] = np.arange(years_of_service) * (1.0 + hydrocarbon_fuel_cagr) ** (
            np.arange(years_of_service) - 1
        )

        partials[
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_rate",
        ] = np.arange(years_of_service) * (1.0 + hydrogen_fuel_cagr) ** (
            np.arange(years_of_service) - 1
        )

        partials[
            "data:cost:operation:electricity_compound_annual_gain_factor",
            "data:cost:operation:electricity_compound_annual_gain_rate",
        ] = np.arange(years_of_service) * (1.0 + electricity_cagr) ** (
            np.arange(years_of_service) - 1
        )


class _AnnualEnergyCostProjection(om.ExplicitComponent):
    """
    Computation of the annual energy cost projection for each energy source over the service life of
    the aircraft. The annual energy cost projection is calculated by multiplying the annual energy
    cost by the compound annual gain factor for each energy source.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            name="data:cost:operation:annual_hydrocarbon_fuel_cost",
            val=0.0,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:annual_hydrogen_fuel_cost",
            val=0.0,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:annual_electricity_cost",
            val=0.0,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
            val=np.nan,
            shape=years_of_service,
        )
        self.add_input(
            name="data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
            val=np.nan,
            shape=years_of_service,
        )
        self.add_input(
            name="data:cost:operation:electricity_compound_annual_gain_factor",
            val=np.nan,
            shape=years_of_service,
        )

        self.add_output(
            "data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            units="USD/yr",
            val=0.0,
            shape=years_of_service,
        )
        self.add_output(
            "data:cost:operation:annual_hydrogen_fuel_cost_projection",
            units="USD/yr",
            val=0.0,
            shape=years_of_service,
        )
        self.add_output(
            "data:cost:operation:annual_electricity_cost_projection",
            units="USD/yr",
            val=0.0,
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            "data:cost:operation:annual_hydrocarbon_fuel_cost",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )
        self.declare_partials(
            "data:cost:operation:annual_hydrogen_fuel_cost_projection",
            "data:cost:operation:annual_hydrogen_fuel_cost",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )
        self.declare_partials(
            "data:cost:operation:annual_electricity_cost_projection",
            "data:cost:operation:annual_electricity_cost",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )
        self.declare_partials(
            "data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
        )
        self.declare_partials(
            "data:cost:operation:annual_hydrogen_fuel_cost_projection",
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
        )
        self.declare_partials(
            "data:cost:operation:annual_electricity_cost_projection",
            "data:cost:operation:electricity_compound_annual_gain_factor",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_hydrocarbon_fuel_cost_projection"] = (
            inputs["data:cost:operation:annual_hydrocarbon_fuel_cost"]
            * inputs["data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor"]
        )
        outputs["data:cost:operation:annual_hydrogen_fuel_cost_projection"] = (
            inputs["data:cost:operation:annual_hydrogen_fuel_cost"]
            * inputs["data:cost:operation:hydrogen_fuel_compound_annual_gain_factor"]
        )
        outputs["data:cost:operation:annual_electricity_cost_projection"] = (
            inputs["data:cost:operation:annual_electricity_cost"]
            * inputs["data:cost:operation:electricity_compound_annual_gain_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        partials[
            "data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            "data:cost:operation:annual_hydrocarbon_fuel_cost",
        ] = inputs["data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor"]

        partials[
            "data:cost:operation:annual_hydrogen_fuel_cost_projection",
            "data:cost:operation:annual_hydrogen_fuel_cost",
        ] = inputs["data:cost:operation:hydrogen_fuel_compound_annual_gain_factor"]

        partials[
            "data:cost:operation:annual_electricity_cost_projection",
            "data:cost:operation:annual_electricity_cost",
        ] = inputs["data:cost:operation:electricity_compound_annual_gain_factor"]

        partials[
            "data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            "data:cost:operation:hydrocarbon_fuel_compound_annual_gain_factor",
        ] = inputs["data:cost:operation:annual_hydrocarbon_fuel_cost"] * np.ones(years_of_service)

        partials[
            "data:cost:operation:annual_hydrogen_fuel_cost_projection",
            "data:cost:operation:hydrogen_fuel_compound_annual_gain_factor",
        ] = inputs["data:cost:operation:annual_hydrogen_fuel_cost"] * np.ones(years_of_service)

        partials[
            "data:cost:operation:annual_electricity_cost_projection",
            "data:cost:operation:electricity_compound_annual_gain_factor",
        ] = inputs["data:cost:operation:annual_electricity_cost"] * np.ones(years_of_service)


class _EnergyCostProjectionSum(om.ExplicitComponent):
    """
    The sum of the annual energy cost projection for each energy source over the service life of the
    aircraft.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            name="data:cost:operation:annual_hydrocarbon_fuel_cost_projection",
            val=np.nan,
            units="USD/yr",
            shape=years_of_service,
        )
        self.add_input(
            name="data:cost:operation:annual_hydrogen_fuel_cost_projection",
            val=np.nan,
            units="USD/yr",
            shape=years_of_service,
        )
        self.add_input(
            name="data:cost:operation:annual_electricity_cost_projection",
            val=np.nan,
            units="USD/yr",
            shape=years_of_service,
        )

        self.add_output(
            name="data:cost:operation:annual_energy_cost_projection",
            val=0.0,
            units="USD/yr",
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_energy_cost_projection"] = (
            inputs["data:cost:operation:annual_hydrocarbon_fuel_cost_projection"]
            + inputs["data:cost:operation:annual_hydrogen_fuel_cost_projection"]
            + inputs["data:cost:operation:annual_electricity_cost_projection"]
        )
