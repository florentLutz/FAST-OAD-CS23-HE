# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..simple_energy_impact import SimpleEnergyImpacts

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "data.xml"


def test_impact_sizing_jet_fuel():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SimpleEnergyImpacts(mission="design")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SimpleEnergyImpacts(mission="design"), ivc)
    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(3558.98, abs=1e-2)
    # The value correspond to the sizing mission of the K100 with approx 6 pax and 1100 nm of range
    # it equates to around 300gCO2/PAX/km
    assert problem.get_val(
        "data:environmental_impact:sizing:energy_emissions", units="kg"
    ) == pytest.approx(0.00, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(
        problem.get_val("data:environmental_impact:sizing:emissions", units="kg"), abs=1e-2
    )
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        2.72, abs=1e-2
    )

    problem.check_partials(compact_print=True)


def test_impact_operational_biofuel():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SimpleEnergyImpacts(mission="operational", fuel_type="biofuel_ft_pathway")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SimpleEnergyImpacts(mission="operational", fuel_type="biofuel_ft_pathway"), ivc
    )
    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(74.37, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:energy_emissions", units="kg"
    ) == pytest.approx(41.8752, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(116.248, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.275, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_impact_both():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SimpleEnergyImpacts(
                mission="both", fuel_type="biofuel_hefa_pathway", electricity_mix="france"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SimpleEnergyImpacts(
            mission="both", fuel_type="biofuel_hefa_pathway", electricity_mix="france"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:environmental_impact:sizing:fuel_emissions", units="kg"
    ) == pytest.approx(866.67, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:energy_emissions", units="kg"
    ) == pytest.approx(0.00, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:sizing:emissions", units="kg"
    ) == pytest.approx(866.67, abs=1e-2)
    assert problem.get_val("data:environmental_impact:sizing:emission_factor") == pytest.approx(
        0.66, abs=1e-2
    )

    assert problem.get_val(
        "data:environmental_impact:operational:fuel_emissions", units="kg"
    ) == pytest.approx(208.63, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:energy_emissions", units="kg"
    ) == pytest.approx(17.05, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emissions", units="kg"
    ) == pytest.approx(225.68, abs=1e-2)
    assert problem.get_val(
        "data:environmental_impact:operational:emission_factor"
    ) == pytest.approx(0.5344, abs=1e-2)

    problem.check_partials(compact_print=True)
