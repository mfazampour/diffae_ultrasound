import argparse
import os
import subprocess

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion
from scipy.ndimage.measurements import label
import torchio as tio
from torchio.transforms import Pad

from sklearn.neighbors import KDTree
import numpy as np

def find_nearest_in_B(A, B):
    # Build a KDTree with points in B
    tree = KDTree(B)

    # For each point in A, find the nearest neighbor in B
    nearest_points = []
    nearest_dists = []
    for point in A:
        dist, ind = tree.query_set([point], k=1)  # returns distance and index of the nearest point
        nearest_points.append(B[ind[0][0]])
        nearest_dists.append(dist)

    return nearest_points, nearest_dists


def voxel_to_world(indices, affine):
    # Convert indices to homogeneous coordinates
    homogeneous_indices = np.concatenate([indices, np.ones((indices.shape[0], 1))], axis=1)

    # Transform indices to world coordinates
    world_coords = np.dot(homogeneous_indices, affine.T)

    return world_coords[:, :3]  # Discard the homogeneous coordinate

###
# Find the border of the foreground in the image
# image_data: 3D numpy array
# foreground_threshold: The threshold to consider a voxel as foreground
#                       (default: 9)
# return: A list of 3D coordinates of the border voxels
###
def find_border(image_data, background_value=9):
    border_positions = []
    for i in range(image_data.shape[0]):
        found_borders = np.zeros(image_data.shape[2])
        for j in range(image_data.shape[1]):
            pos_ = np.where(image_data[i, j, :] != background_value)[0]
            if pos_.shape[0] == 0:
                continue
            border_positions += [[image_data.shape[0] - i, image_data.shape[1] - j, p] for p in pos_ if found_borders[p] == 0]
            found_borders[pos_] = 1
    for i in range(image_data.shape[0]):
        found_borders = np.zeros(image_data.shape[2])
        for j in range(image_data.shape[1] - 1, 0, -1):
            pos_ = np.where(image_data[i, j, :] != background_value)[0]
            if pos_.shape[0] == 0:
                continue
            border_positions += [[image_data.shape[0] - i, image_data.shape[1] - j, p] for p in pos_ if found_borders[p] == 0]
            found_borders[pos_] = 1
    return border_positions


def choose_points(A, th1, th2, B):

    # Convert A to numpy array if it's not
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    # Choose a1 randomly from A
    idx = np.random.choice(A.shape[0])
    a1 = A[idx]
    b1 = B[idx]

    # Calculate distances from a1 to all points in A
    distances_from_a1 = np.linalg.norm(A - a1, axis=1)

    # Get the indices of the points where distance from a1 is between th1 and th2
    indices_a2 = np.where((distances_from_a1 >= th1) & (distances_from_a1 <= th2))

    # If there is at least one point satisfying the condition,
    # choose one from them randomly
    if indices_a2[0].size > 0:
        idx2 = np.random.choice(indices_a2[0])
        a2 = A[idx2]
        b2 = B[idx2]
    else:
        return None, None

    # Now calculate distances from a2 to all points in A
    distances_from_a2 = np.linalg.norm(A - a2, axis=1)

    # Calculate the distance between a1 and a2
    d_a12 = np.linalg.norm(a1 - a2)

    # Get the indices of the points where distance from a2 is between th1 and th2
    # and distance from a1 is between 2*th1 and 2*th2
    indices_a3 = np.where((distances_from_a2 >= th1) & (distances_from_a2 <= th2) &
                          (distances_from_a1 >= 2 * th1) & (distances_from_a1 <= 2 * th2))

    # If there is at least one point satisfying the condition,
    # choose one from them randomly
    if indices_a3[0].size > 0:
        idx3 = np.random.choice(indices_a3[0])
        a3 = A[idx3]
        b3 = B[idx3]
    else:
        return None, None

    return [a1, a2, a3], [b1, b2, b3]

###
# Create a batch file for the given image
# image_path: The path to the image
# output_folder: The folder to save the output files
# batch_file_path: The path to the batch file
# num_samples: The number of samples to create (default: 100)
###
def create_batch_file(image_path, output_folder, batch_file_path, num_samples=100):
    # Load the image
    image = tio.LabelMap(image_path)
    background_value = 9
    padded_image = Pad(10, padding_mode=background_value)(image)
    image_data = padded_image.data.squeeze().numpy()

    # image = nib.load('/Users/farid/Downloads/whole_body_label.nii.gz')
    # image_data = image.get_fdata()
    positions = np.array(find_border(image_data, background_value=background_value))

    # world coordinates
    world_coords = voxel_to_world(positions, padded_image.affine)

    # Randomly choose 1000 rows
    random_rows = np.random.choice(world_coords.shape[0], size=10000, replace=False)

    # Select the chosen rows from the array
    samples = world_coords[random_rows, :]

    # Save the array to a file
    np.savetxt('/tmp/array.txt', samples, delimiter=' ', fmt='%.2f')

    close_samples, nearest_points = find_liver_points(image_data, padded_image, samples)

    # Save the array to a file
    np.savetxt('/tmp/close_points.txt', close_samples, delimiter=' ', fmt='%.2f')

    # trans_splines, dir_splines = [], []

    # Create the batch file and write the header
    with open(batch_file_path, 'w') as file:
        file.write('INPUT; OUTPUT; TRANSSPLINE; DIRSPLINE;\n')

    # define an array of spline points for 100 times
    for i in range(num_samples):
        # define the splines
        [trans_spline, dir_spline] = choose_points(close_samples, th1=10, th2=20, B=nearest_points)
        # if the points are not found, continue
        if trans_spline is None:
            continue

        # define the batch line
        batch_line = f"{image_path}; {os.path.join(output_folder, str(i) + '.imf')}; " \
                     f"{' '.join(map(str, np.array(trans_spline).flatten().tolist()))}; " \
                     f"{' '.join(map(str, np.array(dir_spline).flatten().tolist()))}"

        # write the batch line to the batch file
        with open(batch_file_path, 'a') as file:
            file.write(f'{batch_line}\n')


def find_liver_points(image_data, padded_image, samples):
    # find liver points
    liver = np.where(image_data == 11)
    liver = np.array(liver).T

    # Randomly choose 1000 rows
    random_rows = np.random.choice(liver.shape[0], size=10000, replace=False)
    # Select the chosen rows from the array
    liver_samples = liver[random_rows, :]
    liver[:, 0] = image_data.shape[0] - liver[:, 0]
    liver[:, 1] = image_data.shape[1] - liver[:, 1]
    liver_world = voxel_to_world(liver, padded_image.affine)
    # find close points to liver
    nearest_points, nearest_dists = find_nearest_in_B(samples, liver_world)
    threshold = 20.0  # in mm
    close_samples = samples[np.squeeze(np.array(nearest_dists)) < threshold]
    nearest_points = np.array(nearest_points)[np.squeeze(np.array(nearest_dists)) < threshold]
    return close_samples, nearest_points


if __name__ == '__main__':

    # read arguments using argparse
    parser = argparse.ArgumentParser(description='Generate batch file for registration')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--batch_file_path', type=str, help='Path to the batch file')
    parser.add_argument('--workspace_simulation_file', type=str, help='Path to the workspace simulation file')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    args = parser.parse_args()

    # run the main function using the arguments
    create_batch_file(args.image_path, args.output_folder, args.batch_file_path, args.num_samples)

    # run the batch file using ImFusionConsole using subprocess module given the path to the SimConfig file
    subprocess.run(['ImFusionConsole', args.workspace_simulation_file, f'batch={args.batch_file_path}'])

