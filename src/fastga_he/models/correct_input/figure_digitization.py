"""
Generic class containing all the digitization needed to compute the aerodynamic
coefficient of the aircraft.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import functools
import os.path as pth
from typing import List

import numpy as np
import openmdao.api as om
import pandas as pd
from scipy import interpolate

from . import resources

DELTA_CD_PLAIN_FLAP = "delta_drag_plain_flap.csv"
K_PLAIN_FLAP = "k_plain_flap.csv"
CL_DELTA_TH_PLAIN_FLAP = "cl_delta_th_plain_flap.csv"
K_CL_DELTA_PLAIN_FLAP = "k_cl_delta_plain_flap.csv"

_LOGGER = logging.getLogger(__name__)


class FigureDigitization(om.ExplicitComponent):
    """Provides lift and drag increments due to high-lift devices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def delta_cd_plain_flap(chord_ratio, control_deflection) -> float:
        """
        Roskam data to account for the profile drag increment due to the deployment of plain flap
        (figure 4.44).

        :param chord_ratio: control surface over lifting surface ratio.
        :param control_deflection: control surface deflection, in deg.
        :return delta_cd_flap: profile drag increment due to the deployment of flaps.
        """

        file = pth.join(resources.__path__[0], DELTA_CD_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_15, y_15 = filter_nans(db, ["DELTA_F_15_X", "DELTA_F_15_Y"])
        x_60, y_60 = filter_nans(db, ["DELTA_F_60_X", "DELTA_F_60_Y"])

        if chord_ratio != np.clip(
            chord_ratio, min(min(x_15), min(x_60)), max(max(x_15), max(x_60))
        ):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        x_value_00 = 0.0
        x_value_15 = np.interp(np.clip(float(chord_ratio), min(x_15), max(x_15)), x_15, y_15)
        x_value_60 = np.interp(np.clip(float(chord_ratio), min(x_60), max(x_60)), x_60, y_60)

        if control_deflection != np.clip(control_deflection, 0.0, 60.0):
            _LOGGER.warning(
                "Control surface deflection outside of the range in Roskam's book, value clipped"
            )

        delta_cd_flap = float(
            np.polyval(
                np.polyfit([0.0, 15.0, 60.0], [x_value_00, x_value_15, x_value_60], 2),
                np.clip(control_deflection, 0.0, 60.0),
            )
        )

        return delta_cd_flap

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_prime_plain_flap(flap_angle, chord_ratio):
        """
        Roskam data to estimate the correction factor to estimate non-linear lift behaviour of
        plain flap (figure 8.13).

        :param flap_angle: the flap angle (in °).
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_prime: correction factor to estimate non-linear lift behaviour of plain flap.
        """

        file = pth.join(resources.__path__[0], K_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_10, y_10 = filter_nans(db, ["X_10", "Y_10"])
        x_15, y_15 = filter_nans(db, ["X_15", "Y_15"])
        x_25, y_25 = filter_nans(db, ["X_25", "Y_25"])
        x_30, y_30 = filter_nans(db, ["X_30", "Y_30"])
        x_40, y_40 = filter_nans(db, ["X_40", "Y_40"])
        x_50, y_50 = filter_nans(db, ["X_50", "Y_50"])

        if (
            (flap_angle != np.clip(flap_angle, min(x_10), max(x_10)))
            or (flap_angle != np.clip(flap_angle, min(x_15), max(x_15)))
            or (flap_angle != np.clip(flap_angle, min(x_25), max(x_25)))
            or (flap_angle != np.clip(flap_angle, min(x_30), max(x_30)))
            or (flap_angle != np.clip(flap_angle, min(x_40), max(x_40)))
            or (flap_angle != np.clip(flap_angle, min(x_50), max(x_50)))
        ):
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        k_chord = [
            float(np.interp(np.clip(flap_angle, min(x_10), max(x_10)), x_10, y_10)),
            float(np.interp(np.clip(flap_angle, min(x_15), max(x_15)), x_15, y_15)),
            float(np.interp(np.clip(flap_angle, min(x_25), max(x_25)), x_25, y_25)),
            float(np.interp(np.clip(flap_angle, min(x_30), max(x_30)), x_30, y_30)),
            float(np.interp(np.clip(flap_angle, min(x_40), max(x_40)), x_40, y_40)),
            float(np.interp(np.clip(flap_angle, min(x_50), max(x_50)), x_50, y_50)),
        ]

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k_prime = float(
            np.interp(np.clip(chord_ratio, 0.1, 0.5), [0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_chord)
        )

        return k_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def cl_delta_theory_plain_flap(thickness, chord_ratio):
        """
        Roskam data to estimate the theoretical airfoil lift effectiveness of a plain flap (
        figure 8.14).

        :param thickness: the airfoil thickness.
        :param chord_ratio: flap chord over wing chord ratio.
        :return cl_delta: theoretical airfoil lift effectiveness of the plain flap.
        """

        file = pth.join(resources.__path__[0], CL_DELTA_TH_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_0, y_0 = filter_nans(db, ["X_0", "Y_0"])
        x_04, y_04 = filter_nans(db, ["X_04", "Y_04"])
        x_10, y_10 = filter_nans(db, ["X_10", "Y_10"])
        x_15, y_15 = filter_nans(db, ["X_15", "Y_15"])

        if (
            (chord_ratio != np.clip(chord_ratio, min(x_0), max(x_0)))
            or (chord_ratio != np.clip(chord_ratio, min(x_04), max(x_04)))
            or (chord_ratio != np.clip(chord_ratio, min(x_10), max(x_10)))
            or (chord_ratio != np.clip(chord_ratio, min(x_15), max(x_15)))
        ):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        cld_t = [
            float(np.interp(np.clip(chord_ratio, min(x_0), max(x_0)), x_0, y_0)),
            float(np.interp(np.clip(chord_ratio, min(x_04), max(x_04)), x_04, y_04)),
            float(np.interp(np.clip(chord_ratio, min(x_10), max(x_10)), x_10, y_10)),
            float(np.interp(np.clip(chord_ratio, min(x_15), max(x_15)), x_15, y_15)),
        ]

        if thickness != np.clip(thickness, 0.0, 0.15):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_delta_th = np.interp(np.clip(thickness, 0.0, 0.15), [0.0, 0.04, 0.1, 0.15], cld_t)

        return cl_delta_th

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_cl_delta_plain_flap(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Roskam data to estimate the correction factor to estimate difference from theoretical
        plain flap lift (figure 8.15).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_cl_delta: correction factor to account for difference from theoretical plain
        flap lift.
        """

        file = pth.join(resources.__path__[0], K_CL_DELTA_PLAIN_FLAP)
        db = pd.read_csv(file)

        # Figure 10.64 b
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = filter_nans(db, ["K_CL_ALPHA"])[0]
        k_cl_delta_min_data = filter_nans(db, ["K_CL_DELTA_MIN"])[0]
        k_cl_delta_max_data = filter_nans(db, ["K_CL_DELTA_MAX"])[0]

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        ):
            _LOGGER.warning(
                "Airfoil lift slope ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        k_cl_alpha = np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        )

        k_cl_delta_min = np.interp(k_cl_alpha, k_cl_alpha_data, k_cl_delta_min_data)
        k_cl_delta_max = np.interp(k_cl_alpha, k_cl_alpha_data, k_cl_delta_max_data)

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.05, 0.5)
        k_cl_delta = np.interp(chord_ratio, [0.05, 0.5], [k_cl_delta_min, k_cl_delta_max])

        return k_cl_delta


def filter_nans(database: pd.DataFrame, tags: List[str]) -> List[np.ndarray]:
    """
    Utility function to jointly filter out NaN in the database with the selected tags.
    """

    filtered_db = database[tags].dropna(axis=0, subset=tags)

    database_columns = []
    for tag in tags:
        database_columns.append(filtered_db[tag].to_numpy())

    return database_columns
