import os
import subprocess
import argparse


def save_2d_images(root_folder, destination_folder, imfusion_workspace_path):
    # check root folder for files ending in .imf
    for file in os.listdir(root_folder):
        if file.endswith(".imf"):
            # read the file name without the extension
            file_name = os.path.splitext(file)[0]
            # define the path to the file
            file_path = os.path.join(root_folder, file)
            # create the destination folder
            destination_folder_path = os.path.join(destination_folder, file_name)
            os.makedirs(destination_folder_path, exist_ok=True)
            # call the imfusion console to save the 2d images to the destination folder
            # the params are INPUT and OUTFOLDER for the imfusion console
            subprocess.call(["ImFusionConsole", imfusion_workspace_path, f'INPUT={file_path}',
                             f'OUTFOLDER={destination_folder_path}'])


if __name__ == '__main__':
    # read arguments using argparse
    parser = argparse.ArgumentParser(description='Generate the batch file for the spline generation')
    parser.add_argument('--root_folder', required=True, type=str, help='The root folder of the data')
    parser.add_argument('--destination_folder', required=True, type=str, help='The destination folder of the batch file')
    parser.add_argument('--imfusion_workspace_path', required=True, type=str, help='The path to the imfusion workspace')
    args = parser.parse_args()

    # run the main function using the arguments
    save_2d_images(args.root_folder, args.destination_folder, args.imfusion_workspace_path)




