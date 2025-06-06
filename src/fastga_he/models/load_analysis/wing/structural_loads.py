"""
Computes the structural loads on the wing of the aircraft in the most stringent case according
to aero-structural loads.
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

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .aerostructural_loads import AerostructuralLoadHE, SPAN_MESH_POINT_LOADS
from .constants import HE_SUBMODEL_STRUCTURAL_LOADS


@oad.RegisterSubmodel(
    HE_SUBMODEL_STRUCTURAL_LOADS, "fastga_he.submodel.loads.wings.structural.legacy"
)
class StructuralLoadsHE(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:loads:max_shear:load_factor", val=np.nan)
        self.add_input("data:loads:max_rbm:load_factor", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
            units="m",
        )
        # This add_input is needed because in the other module of the wing load computation,
        # the shape of this vector is copied based on the Y_vector and not having it here would
        # cause the code to crash.
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")

        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:weight:airframe:wing:punctual_mass:y_ratio",
            shape_by_conn=True,
            val=0.0,
        )
        self.add_input(
            "data:weight:airframe:wing:punctual_mass:mass",
            shape_by_conn=True,
            copy_shape="data:weight:airframe:wing:punctual_mass:y_ratio",
            units="kg",
            val=0.0,
        )
        # Same as with punctual loads expect here, we will have a tag to "turn it off" when at MZFW
        self.add_input(
            "data:weight:airframe:wing:punctual_tanks:y_ratio",
            shape_by_conn=True,
            val=0.0,
        )
        self.add_input(
            "data:weight:airframe:wing:punctual_tanks:fuel_inside",
            shape_by_conn=True,
            copy_shape="data:weight:airframe:wing:punctual_tanks:y_ratio",
            units="kg",
            val=0.0,
        )

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")

        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        # Here we add all the inputs necessary for the addition of the distributed mass other
        # than the fuel (batteries for instance), this input will later be an output of the
        # powertrain sizing but their default value will be set at 0 so that it is used the same
        # way as before even when not using the pt file. Note that setting the inputs like that
        # imposes that they are provided as inputs somewhere else as it cannot take default value
        self.add_input(
            "data:weight:airframe:wing:distributed_mass:y_ratio_start",
            shape_by_conn=True,
            val=0.0,
            desc="Array containing the starting positions of all distributed mass on the wing",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_mass:y_ratio_end",
            shape_by_conn=True,
            val=0.0,
            desc="Array containing the end positions of all distributed mass on the wing",
            copy_shape="data:weight:airframe:wing:distributed_mass:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_mass:start_chord",
            shape_by_conn=True,
            val=0.0,
            units="m",
            desc="Array containing the value of the wing chord at the beginning of the distributed mass",
            copy_shape="data:weight:airframe:wing:distributed_mass:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_mass:chord_slope",
            shape_by_conn=True,
            val=0.0,
            desc="Array containing the value of the chord slope for the distributed mass. Mass is assumed to vary with chord only (not thickness)",
            copy_shape="data:weight:airframe:wing:distributed_mass:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_mass:mass",
            shape_by_conn=True,
            val=0.0,
            units="kg",
            desc="Array containing the value of masses that are distributed on the wing",
            copy_shape="data:weight:airframe:wing:distributed_mass:y_ratio_start",
        )
        # Here we add all the inputs necessary for the addition of the distributed tanks
        self.add_input(
            "data:weight:airframe:wing:distributed_tanks:y_ratio_start",
            shape_by_conn=True,
            val=np.nan,
            desc="Array containing the starting positions of all distributed tanks on the wing",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_tanks:y_ratio_end",
            shape_by_conn=True,
            val=np.nan,
            desc="Array containing the end positions of all distributed tanks on the wing",
            copy_shape="data:weight:airframe:wing:distributed_tanks:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_tanks:start_chord",
            shape_by_conn=True,
            val=np.nan,
            units="m",
            desc="Array containing the value of the wing chord at the beginning of the distributed tanks",
            copy_shape="data:weight:airframe:wing:distributed_tanks:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_tanks:chord_slope",
            shape_by_conn=True,
            val=np.nan,
            desc="Array containing the value of the chord slope for the distributed tanks. Fuel mass is assumed to vary with chord only (not thickness)",
            copy_shape="data:weight:airframe:wing:distributed_tanks:y_ratio_start",
        )
        self.add_input(
            "data:weight:airframe:wing:distributed_tanks:fuel_inside",
            shape_by_conn=True,
            val=np.nan,
            units="kg",
            desc="Array containing the value of fuel inside the tanks that are distributed on the wing",
            copy_shape="data:weight:airframe:wing:distributed_tanks:y_ratio_start",
        )

        self.add_output(
            "data:loads:structure:ultimate:force_distribution:wing",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:force_distribution:fuel",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:force_distribution:point_mass",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:force_distribution:distributed_mass",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )

        self.add_output(
            "data:loads:structure:ultimate:shear:wing", units="N", shape=SPAN_MESH_POINT_LOADS
        )
        self.add_output(
            "data:loads:structure:ultimate:shear:fuel", units="N", shape=SPAN_MESH_POINT_LOADS
        )
        self.add_output(
            "data:loads:structure:ultimate:shear:point_mass", units="N", shape=SPAN_MESH_POINT_LOADS
        )
        self.add_output(
            "data:loads:structure:ultimate:shear:distributed_mass",
            units="N",
            shape=SPAN_MESH_POINT_LOADS,
        )

        self.add_output(
            "data:loads:structure:ultimate:root_bending:wing",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:root_bending:fuel",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:root_bending:point_mass",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:root_bending:distributed_mass",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR LOAD COMPUTATION

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]

        semi_span = inputs["data:geometry:wing:span"].item() / 2.0
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]

        load_factor_shear = inputs["data:loads:max_shear:load_factor"]
        load_factor_rbm = inputs["data:loads:max_rbm:load_factor"]
        wing_mass = inputs["data:weight:airframe:wing:mass"]

        # STEP 2/XX - DELETE THE ADDITIONAL ZEROS WE HAD TO PUT TO FIT OPENMDAO AND ADD A POINT
        # AT THE ROOT (Y=0) AND AT THE VERY TIP (Y=SPAN/2) TO GET THE WHOLE SPAN OF THE WING IN
        # THE INTERPOLATION WE WILL DO LATER

        # Reformat the y_vector array as was done in the aerostructural component
        y_vector = AerostructuralLoadHE.delete_additional_zeros(y_vector)
        chord_vector = AerostructuralLoadHE.delete_additional_zeros(chord_vector)
        y_vector, _ = AerostructuralLoadHE.insert_in_sorted_array(y_vector, 0.0)
        chord_vector = np.insert(chord_vector, 0, root_chord)
        y_vector_orig, _ = AerostructuralLoadHE.insert_in_sorted_array(y_vector, semi_span)
        chord_vector_orig = np.append(chord_vector, tip_chord)

        # STEP 3/XX - WE COMPUTE THE BASELINE WEiGHT DISTRIBUTION AND SCALE IT UP ACCORDING TO
        # THE MOST CONSTRAINING CASE IDENTIFIED IN  THE AEROSTRUCTURAL ANALYSIS

        y_vector, point_mass_array_orig = AerostructuralLoadHE.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, 0.0, 0.0, True, False
        )
        _, wing_mass_array_orig = AerostructuralLoadHE.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, wing_mass, 0.0, False, False
        )
        _, fuel_mass_array_orig = AerostructuralLoadHE.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, 0.0, 1.0, False, False
        )
        _, distributed_mass_array_orig = AerostructuralLoadHE.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, 0.0, 0.0, False, True
        )

        point_mass_array = np.maximum(load_factor_shear, load_factor_rbm) * point_mass_array_orig
        wing_mass_array = np.maximum(load_factor_shear, load_factor_rbm) * wing_mass_array_orig
        fuel_mass_array = np.maximum(load_factor_shear, load_factor_rbm) * fuel_mass_array_orig
        distributed_mass_array = (
            np.maximum(load_factor_shear, load_factor_rbm) * distributed_mass_array_orig
        )

        additional_zeros = np.zeros(SPAN_MESH_POINT_LOADS - len(y_vector))
        point_mass_array_outputs = np.concatenate([point_mass_array, additional_zeros])
        wing_mass_array_outputs = np.concatenate([wing_mass_array, additional_zeros])
        fuel_mass_array_outputs = np.concatenate([fuel_mass_array, additional_zeros])
        distributed_mass_array_outputs = np.concatenate([distributed_mass_array, additional_zeros])

        outputs["data:loads:structure:ultimate:force_distribution:wing"] = wing_mass_array_outputs
        outputs["data:loads:structure:ultimate:force_distribution:fuel"] = fuel_mass_array_outputs
        outputs["data:loads:structure:ultimate:force_distribution:point_mass"] = (
            point_mass_array_outputs
        )
        outputs["data:loads:structure:ultimate:force_distribution:distributed_mass"] = (
            distributed_mass_array_outputs
        )

        point_shear_array = AerostructuralLoadHE.compute_shear_diagram(
            y_vector, load_factor_shear * point_mass_array_orig
        )
        wing_shear_array = AerostructuralLoadHE.compute_shear_diagram(
            y_vector, load_factor_shear * wing_mass_array_orig
        )
        fuel_shear_array = AerostructuralLoadHE.compute_shear_diagram(
            y_vector, load_factor_shear * fuel_mass_array_orig
        )
        distributed_shear_array = AerostructuralLoadHE.compute_shear_diagram(
            y_vector, load_factor_shear * distributed_mass_array_orig
        )

        # STEP 4/XX - WE ADD ZEROS AT THE END OF THE RESULT LIFT DISTRIBUTION TO FIT THE FORMAT
        # IMPOSED BY OPENMDAO

        point_shear_array = np.concatenate([point_shear_array, additional_zeros])
        wing_shear_array = np.concatenate([wing_shear_array, additional_zeros])
        fuel_shear_array = np.concatenate([fuel_shear_array, additional_zeros])
        distributed_shear_array = np.concatenate([distributed_shear_array, additional_zeros])

        outputs["data:loads:structure:ultimate:shear:wing"] = wing_shear_array
        outputs["data:loads:structure:ultimate:shear:fuel"] = fuel_shear_array
        outputs["data:loads:structure:ultimate:shear:point_mass"] = point_shear_array
        outputs["data:loads:structure:ultimate:shear:distributed_mass"] = distributed_shear_array

        point_root_bending_array = AerostructuralLoadHE.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * point_mass_array_orig
        )
        wing_root_bending_array = AerostructuralLoadHE.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * wing_mass_array_orig
        )
        fuel_root_bending_array = AerostructuralLoadHE.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * fuel_mass_array_orig
        )
        distributed_root_bending_array = AerostructuralLoadHE.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * distributed_mass_array_orig
        )

        # STEP 4/XX - WE ADD ZEROS AT THE END OF THE RESULT LIFT DISTRIBUTION TO FIT THE FORMAT
        # IMPOSED BY OPENMDAO

        point_root_bending_array = np.concatenate([point_root_bending_array, additional_zeros])
        wing_root_bending_array = np.concatenate([wing_root_bending_array, additional_zeros])
        fuel_root_bending_array = np.concatenate([fuel_root_bending_array, additional_zeros])
        distributed_root_bending_array = np.concatenate(
            [distributed_root_bending_array, additional_zeros]
        )

        outputs["data:loads:structure:ultimate:root_bending:wing"] = wing_root_bending_array
        outputs["data:loads:structure:ultimate:root_bending:fuel"] = fuel_root_bending_array
        outputs["data:loads:structure:ultimate:root_bending:point_mass"] = point_root_bending_array
        outputs["data:loads:structure:ultimate:root_bending:distributed_mass"] = (
            distributed_root_bending_array
        )
