# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

ENDPOINT_NORMALIZATION_FACTOR = {
    "human_health": {
        "climate_change": 7.42e-03,
        "ozone_depletion": 3.19e-05,
        "ionising_radiation": 4.08e-06,
        "particulate_matter_formation": 1.61e-02,
        "photochemical_oxidant_formation_human_health": 1.80e-05,
        "human_toxicity_carcinogenic": 3.42e-05,
        "human_toxicity_non-carcinogenic": 2.08e-04,
        "water_use": 1.96e-04,
    },
    "ecosystem_quality": {
        "climate_change": 2.24e-05 + 6.11e-10,
        "photochemical_oxidant_formation_terrestrial_ecosystems": 2.24e-06,
        "acidification_terrestrial": 8.42e-06,
        "ecotoxicity_terrestrial": 8.19e-04,
        "water_use": 3.48e-06 + 6.16e-10,
        "land_use": 6.23e-04,
        "eutrophication_freshwater": 4.90e-07,
        "ecotoxicity_freshwater": 1.75e-08,
        "ecotoxicity_marine": 4.56e-09,
        "eutrophication_marine": 6.12e-09,
    },
    "resources_quality": {
        "material_resources_metals_minerals": 2.77e04,
        "energy_resources_non-renewablefossil": 2.91e02,
    },
}

MIDPOINT_NORMALIZATION_FACTOR = {  # Taking hierarchic values
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
}

MIDPOINT_TO_ENDPOINT_FACTOR = {
    "human_health": {
        "climate_change": 9.28e-07,
        "ozone_depletion": 5.31e-04,
        "ionising_radiation": 8.5e-09,
        "particulate_matter_formation": 6.29e-04,
        "photochemical_oxidant_formation_human_health": 9.1e-07,
        "human_toxicity_carcinogenic": 3.32e-06,
        "human_toxicity_non-carcinogenic": 2.28e-07,
        "water_use": 2.22e-06,
    },
    "ecosystem_quality": {
        "climate_change": 2.80e-09 + 7.65e-14,
        "photochemical_oxidant_formation_terrestrial_ecosystems": 1.29e-07,
        "acidification_terrestrial": 2.12e-07,
        "ecotoxicity_terrestrial": 1.14e-11,
        "water_use": 1.35e-08 + 6.04e-13,
        "land_use": 8.88e-09,
        "eutrophication_freshwater": 6.71072608705938e-07,
        "ecotoxicity_freshwater": 6.95e-10,
        "ecotoxicity_marine": 1.05e-10,
        "eutrophication_marine": 1.7e-09,
    },
    "resources_quality": {
        "material_resources_metals_minerals": 0.231084371631101,
        "energy_resources_non-renewablefossil": 0.456567536413144,
    },
}

MIDPOINT_IMPACT = {
    "acidification_terrestrial": 0.00038269717076460215,
    "climate_change": 0.19299259793331988,
    "ecotoxicity_freshwater": 0.00053837149243919,
    "ecotoxicity_marine": 0.0008352320570424439,
    "ecotoxicity_terrestrial": 0.09464831363459997,
    "energy_resources_non-renewablefossil": 0.057401088043030866,  # Taking the value for crude oil
    "eutrophication_freshwater": 2.716149753134672e-06,
    "eutrophication_marine": 3.6092031757171488e-06,
    "human_toxicity_carcinogenic": 0.0016235518734284316,
    "human_toxicity_non-carcinogenic": 0.01327681856284311,
    "ionising_radiation": 0.0009557747071045647,
    "land_use": 0.0004314434254718288,
    "material_resources_metals_minerals": 0.0008457920937301236,
    "ozone_depletion": 1.0697043549358991e-08,
    "particulate_matter_formation": 0.00014785003872498837,
    "photochemical_oxidant_formation_human_health": 0.0006734587533064922,
    "photochemical_oxidant_formation_terrestrial_ecosystems": 0.0007002643949260471,
    "water_use": 7.817899383660468e-05,
}

ENDPOINT_WEIGHTING_FACTOR = {
    "human_health": 0.4,
    "ecosystem_quality": 0.4,
    "resources_quality": 0.2,
}

if __name__ == "__main__":
    midpoint_to_endpoint = {}
    for endpoint in list(ENDPOINT_NORMALIZATION_FACTOR.keys()):
        for midpoint in list(ENDPOINT_NORMALIZATION_FACTOR[endpoint].keys()):
            if midpoint in midpoint_to_endpoint:
                midpoint_to_endpoint[midpoint].append(endpoint)
            else:
                midpoint_to_endpoint[midpoint] = [endpoint]

    midpoint_normalized = {}
    for midpoint in list(MIDPOINT_IMPACT.keys()):
        midpoint_normalized[midpoint] = (
            MIDPOINT_IMPACT[midpoint] / MIDPOINT_NORMALIZATION_FACTOR[midpoint]
        )

    # print(midpoint_normalized)

    endpoint_normalized = {}
    endpoint_normalization_dict = {}
    endpoint_dict = {}
    for endpoint in list(ENDPOINT_NORMALIZATION_FACTOR.keys()):
        endpoint_score = 0.0
        endpoint_normalization = 0.0
        for midpoint in list(ENDPOINT_NORMALIZATION_FACTOR[endpoint].keys()):
            endpoint_score += (
                MIDPOINT_IMPACT[midpoint] * MIDPOINT_TO_ENDPOINT_FACTOR[endpoint][midpoint]
            )
            endpoint_normalization += ENDPOINT_NORMALIZATION_FACTOR[endpoint][midpoint]
        endpoint_dict[endpoint] = endpoint_score
        endpoint_score /= endpoint_normalization
        endpoint_normalized[endpoint] = endpoint_score
        endpoint_normalization_dict[endpoint] = endpoint_normalization

    print("endpoint_dict", endpoint_dict)
    print("endpoint_normalized", endpoint_normalized)

    # To compute the equivalent weighting the following formula will be used:
    # For all midpoint impacts (in the resources/recipe_methods.yml file):
    #   Multiply the normalization factor of that midpoint impact by
    #   The sum, for each endpoint impact the midpoint impact contributes to
    #       The weighting factor of that endpoint impact multiplied by
    #       The midpoint to endpoint factor divided by
    #       The normalization factor of that endpoint impact

    weighting_factor_midpoint = {}
    for midpoint in list(MIDPOINT_NORMALIZATION_FACTOR.keys()):
        weighting_factor = 0
        for endpoint in midpoint_to_endpoint[midpoint]:
            weighting_factor += (
                MIDPOINT_NORMALIZATION_FACTOR[midpoint]
                * ENDPOINT_WEIGHTING_FACTOR[endpoint]
                * MIDPOINT_TO_ENDPOINT_FACTOR[endpoint][midpoint]
                / endpoint_normalization_dict[endpoint]
            )
        weighting_factor_midpoint[midpoint] = weighting_factor

    for midpoint in list(weighting_factor_midpoint.keys()):
        print(midpoint, np.format_float_scientific(weighting_factor_midpoint[midpoint], 3))

    single_score = 0.0
    total_weighting = 0.0
    for midpoint in list(midpoint_normalized.keys()):
        single_score += midpoint_normalized[midpoint] * weighting_factor_midpoint[midpoint]
        total_weighting += weighting_factor_midpoint[midpoint]

    print("single_score", single_score)
    print("total_weighting", total_weighting)
