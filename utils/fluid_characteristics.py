import numpy as np
import openmdao.api as om
from numbers import Number
from typing import Sequence, Union
from CoolProp.CoolProp import PropsSI
from scipy.constants import R, atmosphere, foot
from stdatm import Atmosphere


class FluidCharacteristics:
    """
    Computes fluid characteristics used in heat transfer models.

    Usage:
        FluidCharacteristics(temperature, pressure, coolant)
    > temperature: float
    > pressure: float
    > coolant: string

    """

    def __init__(
        self,
        temperature: Union[float, Sequence[float]],
        pressure: Union[float, Sequence[float]],
        coolant: str = None,
    ):
        """
        :param temperature: fluid temperature (K)
        :param pressure: fluid pressure (Pa)
        :param coolant: options include air, water, hydrogen
        """
        self.temperature = temperature
        self.pressure = pressure
        self.coolant = coolant

        # Floats will be provided as output if altitude is a scalar
        self._float_expected = isinstance(temperature, Number)
        self._float_expected = isinstance(pressure, Number)

        # Outputs
        self._density = None
        self._specific_heat_capacity = None
        self._thermal_conductivity = None
        self._dynamic_viscosity = None
        self._Prandtl = None
        self._specific_volume = None

    ## Inputs

    @property
    def temperature(self) -> Union[float, Sequence[float]]:
        """Temperature of fluid."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: Union[float, Sequence[float]]):
        self._temperature = np.asarray(value)

    @property
    def pressure(self) -> Union[float, Sequence[float]]:
        """Pressure in Pa"""
        return self._pressure

    @pressure.setter
    def pressure(self, value: Union[float, Sequence[float]]):
        self._pressure = np.asarray(value)

    @property
    def coolant(self) -> str:
        """Coolant choice"""
        return self._coolant

    @coolant.setter
    def coolant(self, new_coolant):
        if type(new_coolant) == str:  # type checking for name property
            self._coolant = new_coolant
        else:
            raise Exception("Invalid value for coolant")

    ## Outputs

    @property
    def density(self) -> Union[float, Sequence[float]]:
        """Density in kg/m3"""
        if self._density is None:
            self._density = np.zeros(self._temperature.shape)
            if self._coolant == "air":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "air"
                )

            elif self._coolant == "water":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "water"
                )

            elif self._coolant == "hydrogen":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "hydrogen"
                )

            elif self._coolant == "ammonia":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "ammonia"
                )

            elif self._coolant == "ethylene glycol":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "INCOMP::MEG-50%"
                )

            elif self._coolant == "propylene glycol":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "INCOMP::MPG-50%"
                )

            elif self._coolant == "potassium formate":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "INCOMP::MKF-40%"
                )

            elif self._coolant == "R134a":
                self._density = PropsSI(
                    "D", "T", self.temperature, "P", self.pressure, "R134a"
                )

        return self._return_value(self._density)

    @property
    def specific_heat_capacity(self) -> Union[float, Sequence[float]]:
        """Specific heat capacity"""
        if self._specific_heat_capacity is None:
            self._specific_heat_capacity = np.zeros(self._temperature.shape)
            if self._coolant == "air":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "air"
                )

            elif self._coolant == "water":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "water"
                )

            elif self._coolant == "hydrogen":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "hydrogen"
                )

            elif self._coolant == "ammonia":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "ammonia"
                )

            elif self._coolant == "ethylene glycol":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "INCOMP::MEG-50%"
                )

            elif self._coolant == "propylene glycol":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "INCOMP::MPG-50%"
                )

            elif self._coolant == "potassium formate":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "INCOMP::MKF-40%"
                )

            elif self._coolant == "R134a":
                self._specific_heat_capacity = PropsSI(
                    "C", "T", self.temperature, "P", self.pressure, "R134a"
                )

        return self._return_value(self._specific_heat_capacity)

    @property
    def thermal_conductivity(self) -> Union[float, Sequence[float]]:
        """Thermal conductivity"""
        if self._thermal_conductivity is None:
            self._thermal_conductivity = np.zeros(self._temperature.shape)
            if self._coolant == "air":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY", "T", self.temperature, "P", self.pressure, "air"
                )

            elif self._coolant == "water":
                self._specific_heat_capacity = PropsSI(
                    "CONDUCTIVITY", "T", self.temperature, "P", self.pressure, "water"
                )

            elif self._coolant == "hydrogen":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "hydrogen",
                )

            elif self._coolant == "ammonia":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY", "T", self.temperature, "P", self.pressure, "ammonia"
                )

            elif self._coolant == "ethylene glycol":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MEG-50%",
                )

            elif self._coolant == "propylene glycol":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MPG-50%",
                )

            elif self._coolant == "potassium formate":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MKF-40%",
                )

            elif self._coolant == "R134a":
                self._thermal_conductivity = PropsSI(
                    "CONDUCTIVITY", "T", self.temperature, "P", self.pressure, "R134a"
                )

        return self._return_value(self._thermal_conductivity)

    @property
    def dynamic_viscosity(self) -> Union[float, Sequence[float]]:
        """Dynamic viscosity"""
        if self._dynamic_viscosity is None:
            self._dynamic_viscosity = np.zeros(self._temperature.shape)
            if self._coolant == "air":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "air"
                )

            elif self._coolant == "water":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "water"
                )

            elif self._coolant == "hydrogen":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "hydrogen"
                )

            elif self._coolant == "ammonia":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "ammonia"
                )

            elif self._coolant == "ethylene glycol":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "INCOMP::MEG-50%"
                )

            elif self._coolant == "propylene glycol":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "INCOMP::MPG-50%"
                )

            elif self._coolant == "potassium formate":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "INCOMP::MKF-40%"
                )

            elif self._coolant == "R134a":
                self._dynamic_viscosity = PropsSI(
                    "V", "T", self.temperature, "P", self.pressure, "R134a"
                )

        return self._return_value(self._dynamic_viscosity)

    @property
    def Prandtl(self) -> Union[float, Sequence[float]]:
        """Prandtl number"""
        if self._Prandtl is None:
            self._Prandtl = np.zeros(self._temperature.shape)
            if self._coolant == "air":
                self._Prandtl = PropsSI(
                    "PRANDTL", "T", self.temperature, "P", self.pressure, "air"
                )

            elif self._coolant == "water":
                self._Prandtl = PropsSI(
                    "PRANDTL", "T", self.temperature, "P", self.pressure, "water"
                )

            elif self._coolant == "hydrogen":
                self._Prandtl = PropsSI(
                    "PRANDTL", "T", self.temperature, "P", self.pressure, "hydrogen"
                )

            elif self._coolant == "ammonia":
                self._Prandtl = PropsSI(
                    "PRANDTL", "T", self.temperature, "P", self.pressure, "ammonia"
                )

            elif self._coolant == "ethylene glycol":
                self._Prandtl = PropsSI(
                    "PRANDTL",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MEG-50%",
                )

            elif self._coolant == "propylene glycol":
                self._Prandtl = PropsSI(
                    "PRANDTL",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MPG-50%",
                )

            elif self._coolant == "potassium formate":
                self._Prandtl = PropsSI(
                    "PRANDTL",
                    "T",
                    self.temperature,
                    "P",
                    self.pressure,
                    "INCOMP::MKF-40%",
                )

            elif self._coolant == "R134a":
                self._Prandtl = PropsSI(
                    "PRANDTL", "T", self.temperature, "P", self.pressure, "R134a"
                )

        return self._return_value(self._Prandtl)

    @property
    def specific_volume(self) -> Union[float, Sequence[float]]:
        """Specific volume"""
        if self._specific_volume is None:
            self._specific_volume = np.zeros(self._temperature.shape)
            self._specific_volume = 1 / self.density
        return self._return_value(self._specific_volume)

    def _return_value(self, value):
        """
        :returns: a float when needed. Otherwise, returns the value itself.
        """
        if self._float_expected and value is not None:
            try:
                # It's faster to try... catch than to test np.size(value).
                # (but float(value) is slow to fail if value is None, so
                #  it is why we test it before)
                return float(value)
            except TypeError:
                pass
        return value
