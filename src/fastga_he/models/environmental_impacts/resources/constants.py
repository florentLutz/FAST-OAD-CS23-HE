# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

LCA_PREFIX = "data:environmental_impact:"
METHODS_TO_FILE = {
    "EF v3.1 no LT": "ef_no_lt_methods.yml",
    "EF v3.1": "ef_methods.yml",
    "IMPACT World+ v2.0.1": "impact_world_methods.yml",
    "ReCiPe 2016 v1.03": "recipe_methods.yml",
}
METHODS_TO_NORMALIZATION = {
    "EF v3.1 no LT": True,
    "EF v3.1": True,
    "IMPACT World+ v2.0.1": False,  # In EcoInvent, they implement a version that doesn't handle normalisation
    "ReCiPe 2016 v1.03": True,
}
METHODS_TO_WEIGHTING = {
    "EF v3.1 no LT": True,
    "EF v3.1": True,
    "IMPACT World+ v2.0.1": False,  # In EcoInvent, they implement a version that doesn't handle normalisation
    "ReCiPe 2016 v1.03": True,
}
NORMALIZATION_FACTOR = {
    "EF v3.1 no LT": {
        "acidification": 5.56e01,
        "climate_change": 7.55e03,
        "ecotoxicity_freshwater": 5.67e04,
        "energy_resources_non-renewable": 6.50e04,
        "eutrophication_freshwater": 1.61e00,
        "eutrophication_marine": 1.95e01,
        "eutrophication_terrestrial": 1.77e02,
        "human_toxicity_carcinogenic": 1.73e-05,
        "human_toxicity_carcinogenicinorganics": 1.73e-05,  # Same as human_toxicity_carcinogenic
        "human_toxicity_carcinogenicorganics": 1.73e-05,  # Same as human_toxicity_carcinogenic
        "human_toxicity_non-carcinogenic": 1.29e-04,
        "human_toxicity_non-carcinogenicinorganics": 1.29e-04,  # human_toxicity_non-carcinogenic
        "human_toxicity_non-carcinogenicorganics": 1.29e-04,  # human_toxicity_non-carcinogenic
        "ionising_radiation": 4.22e03,
        "land_use": 8.19e05,
        "material_resources_metals_minerals": 6.36e-02,
        "ozone_depletion": 5.23e-02,
        "particulate_matter_formation": 5.95e-04,
        "photochemical_oxidant_formation_human_health": 4.09e01,
        "water_use": 1.15e04,
    },
    "EF v3.1": {
        "acidification": 5.56e01,
        "climate_change": 7.55e03,
        "climate_change_biogenic": 7.55e03,  # Same as climate_change
        "climate_change_fossil": 7.55e03,  # Same as climate_change
        "climate_change_land_use_and_land_use_change": 7.55e03,  # Same as climate_change
        "ecotoxicity_freshwater": 5.67e04,
        "ecotoxicity_freshwaterinorganics": 5.67e04,  # Same as ecotoxicity_freshwater
        "ecotoxicity_freshwaterorganics": 5.67e04,  # Same as ecotoxicity_freshwater
        "energy_resources_non-renewable": 6.50e04,
        "eutrophication_freshwater": 1.61e00,
        "eutrophication_marine": 1.95e01,
        "eutrophication_terrestrial": 1.77e02,
        "human_toxicity_carcinogenic": 1.73e-05,
        "human_toxicity_carcinogenicinorganics": 1.73e-05,  # Same as human_toxicity_carcinogenic
        "human_toxicity_carcinogenicorganics": 1.73e-05,  # Same as human_toxicity_carcinogenic
        "human_toxicity_non-carcinogenic": 1.29e-04,
        "human_toxicity_non-carcinogenicinorganics": 1.29e-04,  # human_toxicity_non-carcinogenic
        "human_toxicity_non-carcinogenicorganics": 1.29e-04,  # human_toxicity_non-carcinogenic
        "ionising_radiation_human_health": 4.22e03,
        "land_use": 8.19e05,
        "material_resources_metals_minerals": 6.36e-02,
        "ozone_depletion": 5.23e-02,
        "particulate_matter_formation": 5.95e-04,
        "photochemical_oxidant_formation_human_health": 4.09e01,
        "water_use": 1.15e04,
    },
    "ReCiPe 2016 v1.03": {  # Taking hierarchic values
        "acidification_terrestrial": 4.10e01,
        "climate_change": 7.99e03,
        "ecotoxicity_freshwater": 2.52e01,
        "ecotoxicity_marine": 4.34e01,
        "ecotoxicity_terrestrial": 1.52e04,
        "energy_resources_non-renewablefossil": 569.90,  # Taking the value for crude oil
        "eutrophication_freshwater": 6.5e-01,
        "eutrophication_marine": 4.62e00,
        "human_toxicity_carcinogenic": 1.03e01,
        "human_toxicity_non-carcinogenic": 3.13e04,
        "ionising_radiation": 4.80e02,
        "land_use": 6.17e03,
        "material_resources_metals_minerals": 1.2e05,
        "ozone_depletion": 6.00e-02,
        "particulate_matter_formation": 2.56e01,
        "photochemical_oxidant_formation_human_health": 2.06e01,
        "photochemical_oxidant_formation_terrestrial_ecosystems": 1.77e01,
        "water_use": 2.67e02,
        "total_human_health": 2.40e-02,
        "total_ecosystem_quality": 1.48e-03,
        "total_natural_resources": 2.80e04,
    },
}

WEIGHTING_FACTOR = {
    "EF v3.1 no LT": {
        "acidification": 0.062,
        "climate_change": 0.2106,
        "ecotoxicity_freshwater": 0.0192,
        "energy_resources_non-renewable": 0.0832,
        "eutrophication_freshwater": 0.0280,
        "eutrophication_marine": 0.0296,
        "eutrophication_terrestrial": 0.0371,
        "human_toxicity_carcinogenic": 0.0213,
        "human_toxicity_non-carcinogenic": 0.0184,
        "ionising_radiation": 0.0501,
        "land_use": 0.0794,
        "material_resources_metals_minerals": 0.0755,
        "ozone_depletion": 0.0631,
        "particulate_matter_formation": 0.0896,
        "photochemical_oxidant_formation_human_health": 0.0478,
        "water_use": 0.0851,
    },
    "EF v3.1": {
        "acidification": 0.062,
        "climate_change": 0.2106,
        "ecotoxicity_freshwater": 0.0192,
        "energy_resources_non-renewable": 0.0832,
        "eutrophication_freshwater": 0.0280,
        "eutrophication_marine": 0.0296,
        "eutrophication_terrestrial": 0.0371,
        "human_toxicity_carcinogenic": 0.0213,
        "human_toxicity_non-carcinogenic": 0.0184,
        "ionising_radiation": 0.0501,
        "land_use": 0.0794,
        "material_resources_metals_minerals": 0.0755,
        "ozone_depletion": 0.0631,
        "particulate_matter_formation": 0.0896,
        "photochemical_oxidant_formation_human_health": 0.0478,
        "water_use": 0.0851,
    },
    # The following weighting factors have been calculated to compute a single score based on
    # midpoints impacts while still keeping the 40/40/20 approach commonly used on endpoints. See
    # in the methodology folder for the computation of those equivalent weighting factors
    "ReCiPe 2016 v1.03": {  # Taking hierarchic values
        "midpoint": {
            "acidification_terrestrial": 2.351e-03,
            "climate_change": 1.296e-01,
            "ecotoxicity_freshwater": 4.737e-06,
            "ecotoxicity_marine": 1.232e-06,
            "ecotoxicity_terrestrial": 4.686e-05,
            "energy_resources_non-renewablefossil": 1.859e-03,  # Taking the value for crude oil
            "eutrophication_freshwater": 1.18e-04,
            "eutrophication_marine": 2.124e-06,
            "human_toxicity_carcinogenic": 5.696e-04,
            "human_toxicity_non-carcinogenic": 1.189e-01,
            "ionising_radiation": 6.797e-05,
            "land_use": 1.482e-02,
            "material_resources_metals_minerals": 1.981e-01,
            "ozone_depletion": 5.307e-04,
            "particulate_matter_formation": 2.682e-01,
            "photochemical_oxidant_formation_human_health": 3.123e-04,
            "photochemical_oxidant_formation_terrestrial_ecosystems": 6.175e-04,
            "water_use": 1.085e-02,
        },
        "endpoint": {
            "total_ecosystem_quality": 4.0e-1,
            "total_human_health": 4.0e-1,
            "total_natural_resources": 2.0e-1,
        },
    },
}
