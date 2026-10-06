"""
This file deals with assigning trajectory frames to their closest grid positions and orientations.
"""
from functools import partial
from multiprocessing import Pool

from MDAnalysis import Universe
from MDAnalysis.lib.distances import calc_bonds
from numpy.typing import NDArray
import numpy as np

def assign_1D(points_to_assign: NDArray, grid_points: NDArray, limits: list, is_periodic: bool) -> NDArray:
    """
    Given a 1D grid (eg. [0, 0.5, 1.0, 1.5]) that might or might not be periodic and 1D points (eg. [0.6, -5.2, 1.7]),
    return an array of indices that tell you which gridpoint is closest to this point.

    When there is no periodicity, this is easy - the gridpoint with the smallest absolute value of difference is the
    closest one.

    If we have a periodic system, the limits play a role, because each gridpoint is infinitely repeated. We use a mod
    operator to get the closest gridpoint.

    Args:
        points_to_assign (NDArray): an array of shape (N_trajlen,) of points to be assigned
        grid_points (NDArray): an array of shape (N_gridpoints,) of 1D gridpoints
        limits (NDArray): a list of two numbers, the lower and upper limit, eg. [0.0, 2.0]
        is_periodic (bool): True if the grid is supposed to be periodic, if False points outside the grid will be
        assigned to the smallest or largest values

    Returns:
        an array of shape (N_trajlen,) where an idex of closest gridpoint is assigned for each point
    """
    N_gridpoints = len(grid_points)
    distance_gridpoints = np.array(np.abs(grid_points[1]-grid_points[0]))

    if is_periodic:
        periodic_box_len = np.abs(limits[1] - limits[0])
        points_to_assign = points_to_assign % periodic_box_len + limits[0]
        assigned_indices = np.floor(points_to_assign / distance_gridpoints + 0.5) % N_gridpoints
    else:
        assigned_indices = np.argmin(np.abs(np.subtract.outer(points_to_assign, grid_points)), axis=1)
    return assigned_indices


def assign_to_cartesian_translation_grid(points_to_assign: NDArray, subgrids: list, subgrid_limits: list,
                                         periodic_in: list) -> tuple:
    """
    How to assign to cartesian grid: assign coordinates individually (x to closest x grid point and so on,
    then add x_index * (N_y_gridpoints * N_z_gridpoints) + y_index * (N_z_gridpoints) + z_index

    Args:
        points_to_assign (NDArray): a (N_points, 3) array of coordinates to be assigned
        subgrids (list): a list of sublists of the form [x_gridpoints, y_gridpoints, z_gridpoints]
        subgrid_limits (list): a list of the form [[x_min, x_max], [y_min, y_max], [z_min, z_max]] - limits are
            important for periodic grids
        periodic_in (list): a list that tells you which dimension is periodic eg [True, True, False]

    Returns:
        a (N_points,) array of translation indices
    """
    assert len(subgrids) == len(subgrid_limits) == len(periodic_in) == 3, f'False statement: {len(subgrids)} == {len(subgrid_limits)} == {len(periodic_in)} == 3'
    N_x, N_y, N_z = [len(subgrid) for subgrid in subgrids]

    xyz_assignments = np.zeros(points_to_assign.shape)

    for i in range(3):
        xyz_assignments[:, i] = assign_1D(points_to_assign[:, i].T, subgrids[i], subgrid_limits[i], periodic_in[i])

    x_assignments, y_assignments, z_assignments = xyz_assignments.T

    combined_assignments = x_assignments*N_y*N_z + y_assignments*N_z + z_assignments
    return x_assignments, y_assignments, z_assignments, combined_assignments


def assign_to_spherical_grid(points_to_assign: NDArray, subgrids: list) -> tuple:
    """
    Assign Cartesian points to a spherical grid: the orientation is assigned to the closest direction
    (largest dot product with the unit vector of the point), the distance to the closest radial grid value.
    Combined index: radial_index * N_o + orientation_index

    Args:
        points_to_assign (NDArray): a (N_points, 3) array of Cartesian coordinates to be assigned
        subgrids (list): a list of the form [orientations, radii], where orientations is a (N_o, 3)
            list/array of unit vectors and radii is a list of N_r radial grid values

    Returns:
        a tuple (orientation_assignments, radial_assignments, combined_assignments), each of shape (N_points,)
    """
    points = np.asarray(points_to_assign, dtype=float)
    orientations = np.asarray(subgrids[0], dtype=float)
    radii = np.asarray(subgrids[1], dtype=float)
    N_o = len(orientations)

    assert len(subgrids) == 2, f'Expected subgrids of the form [orientations, radii], got {len(subgrids)} entries'
    assert points.ndim == 2 and points.shape[1] == 3, f'points_to_assign must have shape (N_points, 3), got {points.shape}'
    assert orientations.ndim == 2 and orientations.shape[1] == 3, f'orientations must have shape (N_o, 3), got {orientations.shape}'
    assert radii.ndim == 1, f'radii must be a 1D list/array, got shape {radii.shape}'

    # radial part: distance to origin, then closest radial grid value
    distances = np.linalg.norm(points, axis=1)
    radial_assignments = np.argmin(np.abs(distances[:, None] - radii[None, :]), axis=1)

    # orientation part: unit vectors (points at the origin get an arbitrary direction, index 0)
    unit_vectors = np.divide(points, distances[:, None], out=np.zeros_like(points), where=distances[:, None] > 0)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)  # in case they're not normalized
    orientation_assignments = np.argmax(unit_vectors @ orientations.T, axis=1)

    combined_assignments = radial_assignments * N_o + orientation_assignments    
    return orientation_assignments, radial_assignments, combined_assignments


## ToDo: orientation should be renamed to rotation
def assign_to_best_orientation(structures_to_assign: NDArray, structure_at_each_gridpoint: NDArray) -> NDArray:
    """
    We want to determine the best rotational fit for the structures in the trajectory among the possibilities given by
    the grid.

    IMPORTANT! The function assumes that all of these structures are centered so that center of mass is at origin.

    Args:
        structures_to_assign (NDArray): an array of shape (N_frames, N_atoms, 3) - each "row" a structure to be assigned
        structure_at_each_gridpoint (NDArray): an array of shape (N_rotations, N_atoms, 3) - each "row" a reference
            structure

    Returns:
        an array of shape (N_frames,), each element an index of best-fitting orientation
    """

    # shape (N_rotations, N_frames)
    distances = np.empty((len(structure_at_each_gridpoint), len(structures_to_assign)))

    # maybe could use worker pool here
    # right now this takes 9min for an extremely long trajectory & 80 reference rotations, so not too worried

    # we loop over reference structures since this is presumably the much shorter list
    # for i, ref_structure in enumerate(structure_at_each_gridpoint):
    #     distances_to_this_ref = np.linalg.norm((structures_to_assign - ref_structure), axis=2)
    #     total_distances = distances_to_this_ref.sum(axis=1)
    #     distances[i] = total_distances
    # best_indices = np.argmin(distances.T, axis=1)

    # faster, since: ||x − r||² = ||x||² − 2x·r + ||r||² (and ||x||² is constant)
    X = structures_to_assign.reshape(len(structures_to_assign), -1)                # (N_frames, 39)
    R = structure_at_each_gridpoint.reshape(len(structure_at_each_gridpoint), -1)  # (N_rot, 39)
    best_indices = np.argmin((R**2).sum(1) - 2 * X @ R.T, axis=1)                  # argmin[axis=1](N_frames, N_rot)

    return best_indices

def _calc_distance_along_traj(frame_index, u, reference_coordinates):
    # index the trajectory to set it to the frame_index frame
    u.trajectory[frame_index]
    distance_between = calc_bonds(u.atoms, reference_coordinates)
    return np.sum(distance_between)


def loop_slowly_over_trajectories(u, u_ref, ag, ag_ref, N_rotations, limit_to):
    """
    Not in use right now because looping with GROMACS is faster than any python-based solution. I leave this function in
    case we later need to loop over a trajectory and calculate something more complex.

    The idea behind having a fast and slow version of the same function: the slow version is more readable and both
    should return the same answer - helps us test the complex fast function.
    """
    all_rmsds = np.zeros((limit_to, N_rotations))
    for ts in u_ref.trajectory[:N_rotations]:
        for traj_ts in u.trajectory[:limit_to]:
            distance_between = calc_bonds(ag,ag_ref)
            distance_between = np.sum(distance_between)
            all_rmsds[traj_ts.frame, ts.frame] = distance_between
    return all_rmsds


def loop_fast_over_trajectories(u, u_ref, N_rotations, limit_to=None):
    """
    Not in use right now because looping with GROMACS is faster than any python-based solution. I leave this function in
    case we later need to loop over a trajectory and calculate something more complex.

    The idea behind having a fast and slow version of the same function: the slow version is more readable and both
    should return the same answer - helps us test the complex fast function.
    """
    if limit_to is None:
        limit_to = len(u.trajectory)
    all_rmsds = np.zeros((limit_to, N_rotations))
    for ts in u_ref.trajectory[:N_rotations]:
        run_per_frame = partial(_calc_distance_along_traj,
            u=u,
            reference_coordinates=u_ref.atoms.positions)
        frame_values = np.arange(limit_to)
        with Pool(3) as worker_pool:
            resulting_distances = worker_pool.map(run_per_frame,frame_values)
        resulting_distances = np.array(resulting_distances)
        all_rmsds[:, ts.frame] = resulting_distances
    return all_rmsds
