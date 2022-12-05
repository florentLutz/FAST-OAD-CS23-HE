"""Module for custom Exception classes."""

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


class FastOadGaHeError(Exception):
    """Base Class for exceptions related to the FAST-OAD-GA-HE framework."""


class ControlParameterInconsistentShapeError(FastOadGaHeError):
    """
    Control Parameter Inconsistent Shape Error.

    This exception is used when the shape of a parameter used for the mission is not consistent.
    It should be equal to 1 (same value for the whole mission), 3 (one value for each phase) or
    number_of_points (one value for each point).
    """
