import argparse

from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # from the arguments, read if it is running on the cluster or not
    parser = argparse.ArgumentParser(description='Generate the batch file for the spline generation')
    parser.add_argument('--cluster', action='store_true', help='If the script is running on the cluster')
    # add argument that specifies the number of gpus
    parser.add_argument('--gpus', type=int, default=1, help='The number of gpus to use')
    # add clahe argument
    parser.add_argument('--clahe', action='store_true', help='If clahe should be used')
    # add compressed argument
    parser.add_argument('--compress', action='store_true', help='If the latent space should be compressed')
    args = parser.parse_args()

    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = args.gpus
    conf = ultrasound_autoenc_cls(args.cluster, use_clahe=args.clahe, compress_latent=args.compress)
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!
