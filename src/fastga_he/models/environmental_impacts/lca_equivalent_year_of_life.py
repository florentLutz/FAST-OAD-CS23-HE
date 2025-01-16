# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om


class LCAEquivalentYearOfLife(om.ExplicitComponent):
    """
    The models have been implemented in a way were the input are the number of year the aircraft
    is expected to operate and the number of flights per year. In practice data rather give the
    time in hours an airframe is expected to live (airframe hours) and the average number of flight
    hours in a year.

    To avoid complicate rework of existing component, we'll simply compute an equivalent of the
    former based on the latter. The default value will be the average for a 1 engine turboprop AC
    as computed based on the data of the GA survey of the FAA.
    """

    def setup(self):
        self.add_input(
            name="data:TLAR:max_airframe_hours",
            val=3524.9,
            units="h",
            desc="Expected lifetime of the aircraft expressed in airframe hours",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            name="data:TLAR:aircraft_lifespan",
            val=20.0,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:TLAR:aircraft_lifespan"] = (
            inputs["data:TLAR:max_airframe_hours"] / inputs["data:TLAR:flight_hours_per_year"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:TLAR:aircraft_lifespan", "data:TLAR:max_airframe_hours"] = (
            1.0 / inputs["data:TLAR:flight_hours_per_year"]
        )
        partials["data:TLAR:aircraft_lifespan", "data:TLAR:flight_hours_per_year"] = (
            -inputs["data:TLAR:max_airframe_hours"]
            / inputs["data:TLAR:flight_hours_per_year"] ** 2.0
        )
