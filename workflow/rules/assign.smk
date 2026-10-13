"""
Every rule here should save to assign/ directory. Assigning is performed on the simulated trajectory and assigns every
frame of that trajectory to the best grid point. In addition, wrapping is performed, where the simulation frames are
wrapped to the smallest cuboid cell, mostly for plotting of periodic structures.
"""

from MDAnalysis import Universe, Writer
import numpy as np
import plotly.graph_objects as go
from MDAnalysis.topology.guessers import guess_masses

import MDAnalysis as mda
from MDAnalysis.transformations import center_in_box 
from MDAnalysis.coordinates.memory import MemoryReader


from molgri.molecules.find_unit_cell import get_cuboid_cell_side_lengths, wrap_multiple_atoms_to_cuboid_cell, wrap_to_cuboid_cell
from molgri.molecules.assignment import assign_to_cartesian_translation_grid, assign_to_spherical_grid, assign_to_best_orientation

from workflow.helpers.io import get_num_atoms, read_object, write_object

##################################### POSITION ASSIGNMENT ####################################################

# Helper rule to resolve the pathvar
rule _sim_root:
    input:
        "<output_specific_molecule>"

rule copy_molecular_files_from_input_2:
    """
    Here the goal is just to start a new directory and copy molecular files there.
    """
    input:
        molecule_1 = f"<inputs_structures><molecule1>.<ext_inp>",
        molecule_2 = f"<inputs_structures><molecule2>.<ext_inp>",
    output:
        molecule_1 = f"<simulation>molecule1.<ext_inp>",
        molecule_2 = f"<simulation>molecule2.<ext_inp>",
    run:
        from molgri.molecules.bimolecular import move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        # center molecules
        m1 = move_to_center(m1)
        m2 = move_to_center(m2)

        write_object(m1, output.molecule_1)
        write_object(m2, output.molecule_2)

rule create_structure_2:
    input:
        molecule_1 = f"<simulation>molecule1.<ext_inp>",
        molecule_2 = f"<simulation>molecule2.<ext_inp>",
    output:
        structure = f"<simulation>structure.<ext_str>",
    run:
        from molgri.molecules.bimolecular import get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        z_distance = float(config["grid"]["translation_subgrids_A"][-1][1])
        structure = get_bimolecular_structure(m1, m2, z_distance=z_distance)
        write_object(structure, output.structure)

# ToDO: The following rule is obsolete and invalid if M1 is NOT centered at (0,0,0). For the assignment, any location of M1 is allowed 
rule wrap_trajectory2cuboid_cell:
    """
    This rule takes a normal (already centered etc.) trajectory of two molecules and applies periodic boundary
    conditions of the cuboid cell to the molecule2. This is not used for assignment but for plotting trajectories.
    
    This is tested - you can look at frame_i of wrapped trajectory and frame_i of simulation to confirm.
    """
    input:
        structure = f"<simulation>structure.<ext_str>",
        trajectory = f"<simulation>trajectory.<ext_trj>", #f"<simulation>trajectory.<ext_trj>", #f"trajectory_new.lammpstrj", #f"<simulation>trajectory.<ext_trj>",
        structure1 = f"<simulation>molecule1.<ext_inp>",
    benchmark:
        "<outputs_assignment>duration_wrapping_trajectory.txt"
    output:
        wrapped_trajectory = f"<outputs_assignment>wrapped_trajectory.<ext_trj>",
    params:
        cuboid_cell_lengths = config["pseudotrajectory"]["cuboid_cell"]
    run:
        n_mol1 = get_num_atoms(input.structure1)
        u = Universe(input.structure, input.trajectory)
        u.add_TopologyAttr('elements')
        u.add_TopologyAttr('names')
        dict_labels = {"1": "H", "2": "C", "3": "O", "4": "O", "5": "O", "6": "Zr", "OW": "O", "HW1": "H", "HW2": "H",
                       "7": "Ar", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CG": "C", "CZ": "C", "HD1": "H",
                       "HD2": "H", "HE1": "H", "HE2": "H", "HG": "H", "HZ": "H", "Z": "Zr", "A": "Ar"}
        #initial_labels = u.atoms.names
        u.atoms.elements = [dict_labels[name] if name in dict_labels.keys() else name for name in u.atoms.types]
        u.atoms.names = [dict_labels[name] if name in dict_labels.keys() else name for name in u.atoms.types]
        u.atoms.types = u.atoms.names
        u.add_TopologyAttr('masses')
        u.atoms.masses = guess_masses(u.atoms.elements)

        # try:
        #     u = Universe(input.trajectory,format="LAMMPSDUMP")
        # except ValueError:
        #     u = Universe(input.structure, input.trajectory,format="XYZ")
        ag2 = u.select_atoms("all")
        ag2 = ag2[n_mol1:]


        side_lengths = np.array(params.cuboid_cell_lengths)# get_cuboid_cell_side_lengths(input.structure1)

        with Writer(output.wrapped_trajectory, u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                ts.positions[n_mol1:] = wrap_multiple_atoms_to_cuboid_cell(ag2.center_of_mass(),
                    ts.positions[n_mol1:],
                    side_lengths,
                    wrap_only_xy=False)
                W.write(u.atoms)

rule wrap_molecule2_COM:
    """
    This rule uses gromacs' output that contains the center of mass of molecule 2 for each frame of the trajectory. This
    center of mass is wrapped to the cuboid cell and later used for position assignment.
    """
    input:
        com_m2 = f"<simulation>COM_m2.xvg",
        structure1 = f"<simulation>molecule1.<ext_inp>",
    output:
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
    params:
        cuboid_cell_lengths = config["pseudotrajectory"]["cuboid_cell"]
    run:
        com_m2 = read_object(input.com_m2)
        # print(com_m2, len(com_m2))
        # NEED TO CONVERT TO ANGSTROM
        com_array_m2 = 10 * com_m2.to_numpy()[:, -3:]
        # print(com_array_m2)

        write_object(com_array_m2, output.com_m2)

        # determine cuboid cell
        side_lengths = np.array(params.cuboid_cell_lengths) #get_cuboid_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2, wrap_only_xy=False)
        write_object(wrapped_com_m2, output.com_m2_wrapped)

rule position_assignment_csv:
    """
    Here the position gridpoint that best fits the wrapped center of mass of molecule 2 is found for each frame.
    
    We also provide a .csv file with more information, very usefula for debugging.
    """
    input:
#        energy_csv = f"<simulation>energy.csv",
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
        grid_info= "<outputs_network>grid_info.yaml",
        indices_csv= f"<outputs_network>indices_interpretation.csv"
    output:
#        assignment_csv =  f"<outputs_assignment>translation_assignment.csv",
        translation_assignment= f"<outputs_assignment>translation_assignment.npy",
    params:
        translation_algorithm = config["grid"]["translation_algorithm"]
    run:

#        import pandas as pd
        import numpy as np

        com_m2 = read_object(input.com_m2)
        com_m2_wrapped = read_object(input.com_m2_wrapped)
        print("len com ", len(com_m2))

#        energy = read_object(input.energy_csv)["Energy [kJ/mol]"]
#        print("len energy ", len(energy.to_numpy()))
#         df_indices = read_object(input.indices_csv, header=[0, 1])

#         df_indices.columns = [col[0] if col[1].startswith("Unnamed") else col[1] for col in df_indices.columns]


#         df_indices = df_indices.loc[df_indices["Rotation index"] == 0]

        grid_info = read_object(input.grid_info)
        subgrids = grid_info["subgrid_points"]
        subgrid_limits = grid_info["subgrid_limits_A"]
        periodic_in = grid_info["periodic_in"]
        N_gridpoints = grid_info["subgrid_N_points"]
        print("N_gridpoints ", N_gridpoints)

        match params.translation_algorithm:
            case "spherical":
                data = assign_to_spherical_grid(com_m2, subgrids)
            case "cartesian":
                data = assign_to_cartesian_translation_grid(com_m2, subgrids, subgrid_limits, periodic_in)
            case _:
                raise ValueError("unknown translation algorithm")

#        assigned_array = data[-1].astype(int).T
#         df_assignments = pd.DataFrame(np.array(data).T, columns=["X index", "Y index", "Z index", "Total position index"], dtype=int)
# 
#         df_assignments[["x_COM", "y_COM", "z_COM"]] = pd.DataFrame(list(map(tuple, np.round(com_m2, 5))),index=df_assignments.index)
#         df_assignments[
#             ["x_wCOM", "y_wCOM", "z_wCOM"]] = pd.DataFrame(list(map(tuple, np.round(com_m2_wrapped, 5))),index=df_assignments.index)
# 
#         rows = (
#             df_indices
#             .set_index("Translation index")
#             .loc[assigned_array]
#             .reset_index()
#         )
# 
#         df_assignments["x"] = rows["x"]
#         df_assignments["y"] = rows["y"]
#         df_assignments["z"] = rows["z"]
# 
# 
#         #df_assignments["Assigned position gridpoint"] = list(map(tuple, np.round(position_array, 5)))
#        df_assignments["Energy [kJ/mol]"] = energy.to_numpy()

        tuples = [
            ("Direction indices", "X_index"),
            ("Direction indices", "Y_index"),
            ("Direction indices", "Z_index"),
            ("Total position index", ""),
            ("Exact COM position", "x_COM"),
            ("Exact COM position", "y_COM"),
            ("Exact COM position", "z_COM"),
            ("Wrapped COM position", "x_wCOM"),
            ("Wrapped COM position", "y_wCOM"),
            ("Wrapped COM position", "z_wCOM"),
            ("Assigned position gridpoint", "x"),
            ("Assigned position gridpoint", "y"),
            ("Assigned position gridpoint", "z"),
            ("Energy [kJ/mol]", ""),
        ]

#        df_assignments.columns = pd.MultiIndex.from_tuples(tuples)
#        write_object(df_assignments, output.assignment_csv)
        write_object(np.array(data[-1]), output.translation_assignment)



##################################### ROTATION ASSIGNMENT ####################################################


# ToDo: change to reference frame of m1 to allow any orientation of m1
rule rotation_assignment:
    """
    The closest rotational fit of molecule 2 is found and reported as rotation index.
    """
    input:
        structure2 = f"<pseudosimulation>molecule2.<ext_inp>",
        trajectory=f"<simulation>m2_trajectory_centered.<ext_trj>",
        reference_pt = f"<pseudosimulation>m2_trajectory_centered.<ext_trj>"
    benchmark:
        repeat(f"<outputs_assignment>duration_rotational_assignment.txt",1)
    output:
        assigned_trajectory = f"<outputs_assignment>rotation_assignment.npy"
    run:
        # the trajectories must only contain m2 and it needs to be already centered at origin
        # NB: works on absolute orientation, i.e. assumes fixed orientation of m1
        trajectory_universe = Universe(input.structure2, input.trajectory)
        reference_universe = Universe(input.structure2, input.reference_pt)
        print("traj ref ", len(trajectory_universe.trajectory), len(reference_universe.trajectory))

        N_rotations = int(config["grid"]["N_rotations"])

        # shape (N_frames, N_atoms, 3)
        pos = np.array([ts.positions.copy() for ts in trajectory_universe.trajectory])
        # shape (N_rotations, N_atoms, 3)
        pos_ref = np.array([ts.positions.copy() for ts in reference_universe.trajectory[:N_rotations]])

        best_indices = assign_to_best_orientation(pos, pos_ref)
        write_object(best_indices, output.assigned_trajectory)

##################################### FULL ASSIGNMENT ####################################################

checkpoint full_assignment:
    """
    We combine the rotation assignment and position assignment into the final, full assignment index.
    """
    input:
        rotation_assignment=f"<outputs_assignment>rotation_assignment.npy",
        translation_assignment=f"<outputs_assignment>translation_assignment.npy",
        grid_info=f"<outputs_network>grid_info.yaml",
    output:
        full_assignment = f"<outputs_assignment>full_assignment.npy",
    run:
        rotation_assignments = read_object(input.rotation_assignment)
        translation_assignments = read_object(input.translation_assignment)
        print("rotation assignments ", rotation_assignments.shape)
        print("translation assignments ",translation_assignments.shape)
        grid_info = read_object(input.grid_info)
        N_rotations = int(grid_info["N_rotations"])
        full_assignments = translation_assignments * N_rotations + rotation_assignments
        write_object(full_assignments, output.full_assignment)


rule count_assignments:
    """
    Some quick plotting to show which states are low in energy (lots of assignments) and which are high (basically no
    assignments).
    """
    input:
        translation_assignment=f"<outputs_assignment>translation_assignment.npy",
        rot_assignment=f"<outputs_assignment>rotation_assignment.npy",
        full_assignment=f"<outputs_assignment>full_assignment.npy"
    output:
        translation_assignment = f"<outputs_other_plots>hist_translation_assignment.png",
        rot_assignment = f"<outputs_other_plots>hist_rotation_assignment.png",
        full_assignment = f"<outputs_other_plots>hist_full_assignment.png",
    run:
        for input_file, output_file in zip(input, output):
            data = read_object(input_file)
            fig = go.Figure(
                go.Histogram(
                    x=data,
                    xbins=dict(
                        start=data.min() - 0.5,
                        end=data.max() + 0.5,
                        size=1)
                )
            )

            fig.write_image(output_file, scale=3)

#
# def get_input_files(wc):
#     if "<simulation>" in wc.path:
#         return {"trajectory": f"{wc.path}dump.lammpstrj", "structure1": f"{wc.path}molecule1.<ext_inp>"}
#     else:
#         return {"trajectory": f"{wc.path}trajectory.<ext_trj>", "structure1": f"{wc.path}molecule1.<ext_inp>", "structure": f"{wc.path}structure.<ext_str>"}
#

rule trajectory_centered_at_m2_COM_MDAnalysis:
    """
    Write the whole trajectory translated in such a way that the COM of molecule 2 is at (0,0,0) in each frame and
    molecule1 is not written. This is useful so we can later assign the best rotation.
    """
    input:
        trajectory="<simulation>trajectory.<ext_trj>",
        structure1 ="<simulation>molecule1.<ext_inp>",
    output:
        trajectory="<simulation>m2_trajectory_centered.<ext_trj>",
    run:
        n_mol1 = get_num_atoms(input.structure1)
        u = read_object(input.trajectory)
        ag2 = u.atoms[n_mol1:]

        n_frames = u.trajectory.n_frames
        coords = np.empty((n_frames, ag2.n_atoms, 3), dtype=np.float32)
        has_box = u.dimensions is not None
        dims = np.empty((n_frames, 6), dtype=np.float32) if has_box else None

        for i, ts in enumerate(u.trajectory):
            coords[i] = ag2.positions
            if has_box:
                dims[i] = ts.dimensions

        # Vectorized COM centering for all frames at once
        m = ag2.masses
        com = np.einsum("fai,a->fi", coords, m) / m.sum()   # (n_frames, 3)
        coords -= com[:, None, :]

        # write_object expects a MDA Universe, so we have to generate a new one
        new_u = mda.Merge(ag2)
        new_u.load_new(coords, format=MemoryReader, dimensions=dims, order="fac")
        write_object(new_u, output.trajectory)

rule trajectory_centered_at_m2_COM_MDAnalysis_pseudo:
    """
    Write the whole trajectory translated in such a way that the COM of molecule 2 is at (0,0,0) in each frame and
    molecule1 is not written. This is useful so we can later assign the best rotation.
    """
    input:
        trajectory="<pseudosimulation>trajectory.<ext_trj>",
        structure1 ="<pseudosimulation>molecule1.<ext_inp>",
        structure="<pseudosimulation>structure.<ext_str>",
    output:
        trajectory="<pseudosimulation>m2_trajectory_centered.<ext_trj>",
    run:
        n_mol1 = get_num_atoms(input.structure1)
        u = read_object(input.trajectory)
        ag2 = u.atoms[n_mol1:]

        n_frames = u.trajectory.n_frames
        coords = np.empty((n_frames, ag2.n_atoms, 3), dtype=np.float32)
        has_box = u.dimensions is not None
        dims = np.empty((n_frames, 6), dtype=np.float32) if has_box else None

        for i, ts in enumerate(u.trajectory):
            coords[i] = ag2.positions
            if has_box:
                dims[i] = ts.dimensions

        # Vectorized COM centering for all frames at once
        m = ag2.masses
        com = np.einsum("fai,a->fi", coords, m) / m.sum()   # (n_frames, 3)
        coords -= com[:, None, :]

        # write_object expects a MDA Universe, so we have to generate a new one
        new_u = mda.Merge(ag2)
        new_u.load_new(coords, format=MemoryReader, dimensions=dims, order="fac")
        write_object(new_u, output.trajectory)

rule m2_COM_MDAnalysis:
    """
    Write the whole trajectory of the COM of molecule 2.
    This is used later for assigning the to the translation grid.
    Removes COM of molecule 1 (no effect if it is already centered at (0,0,0))
    """
    input:
        trajectory=f"<simulation>trajectory.<ext_trj>",
        structure1=f"<simulation>molecule1.<ext_inp>",
    output:
        trajectory="<simulation>COM_m2.xvg",
    run:

        def save_xvg(filename, data, xlabel="Time (ps)", ylabel="Value", title=""):
            with open(filename,"w") as f:
                for i in range(13):
                    f.write("#\n")
                f.write(f'@    title \"{title}\"\n')
                f.write(f'@    xaxis label \"{xlabel}\"\n')
                f.write(f'@    yaxis label \"{ylabel}\"\n')
                f.write(f'@ s0 legend "MOL2 X"\n')
                f.write(f'@ s0 legend "MOL2 Y"\n')
                f.write(f'@ s0 legend "MOL2 Z"\n')
                f.write("@TYPE xy\n")
                np.savetxt(f, data, fmt="%.8f")

        n_mol1 = get_num_atoms(input.structure1)
        u = read_object(input.trajectory)
        ag1 = u.atoms[:n_mol1]
        ag2 = u.atoms[n_mol1:]

        # Output writer
        n_frames = len(u.trajectory)
        data = np.empty((n_frames, 4))
        for i, ts in enumerate(u.trajectory):
            data[i, 0] = ts.time
            # data[i, 1:] = ag2.center_of_mass()
            data[i, 1:] = ag2.center_of_mass() - ag1.center_of_mass()
        
        # Å -> nm
        save_xvg(output.trajectory, data / 10.0, xlabel="Time (ps)", ylabel="Coordinate (nm)", title="Center of mass")


rule m2_COM_MDAnalysis_pseudo:
    """
    Write the whole trajectory of the COM of molecule 2.
    This is used later for assigning the to the translation grid.
    Optional: remove COM of molecule 1 if not already centered at (0,0,0)
    """
    input:
        trajectory=f"<pseudosimulation>trajectory.<ext_trj>",
        structure1=f"<pseudosimulation>molecule1.<ext_inp>",
        structure="<pseudosimulation>structure.<ext_str>"
    output:
        trajectory="<pseudosimulation>COM_m2.xvg",
    run:

        def save_xvg(filename, data, xlabel="Time (ps)", ylabel="Value", title=""):
            with open(filename,"w") as f:
                for i in range(13):
                    f.write("#\n")
                f.write(f'@    title \"{title}\"\n')
                f.write(f'@    xaxis label \"{xlabel}\"\n')
                f.write(f'@    yaxis label \"{ylabel}\"\n')
                f.write(f'@ s0 legend "MOL2 X"\n')
                f.write(f'@ s0 legend "MOL2 Y"\n')
                f.write(f'@ s0 legend "MOL2 Z"\n')
                f.write("@TYPE xy\n")
                np.savetxt(f, data, fmt="%.8f")

        n_mol1 = get_num_atoms(input.structure1)
        u = read_object(input.trajectory)
        ag1 = u.atoms[:n_mol1]
        ag2 = u.atoms[n_mol1:]

        # Output writer
        n_frames = len(u.trajectory)
        data = np.empty((n_frames, 4))
        for i, ts in enumerate(u.trajectory):
            data[i, 0] = ts.time
            # data[i, 1:] = ag2.center_of_mass() / 10.0   # Å -> nm
            data[i, 1:] = (ag2.center_of_mass() - ag1.center_of_mass()) / 10.0   # Å -> nm
        
        save_xvg(output.trajectory, data, xlabel="Time (ps)", ylabel="Coordinate (nm)", title="Center of mass")
