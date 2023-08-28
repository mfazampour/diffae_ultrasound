import argparse

from templates import *
from templates_latent import *

if __name__ == '__main__':

    # from the arguments, read if it is running on the cluster or not
    parser = argparse.ArgumentParser(description='Generate the batch file for the spline generation')
    parser.add_argument('--cluster', action='store_true', help='If the script is running on the cluster')
    # add argument that specifies the number of gpus
    parser.add_argument('--gpus', type=int, default=1, help='The number of gpus to use')
    # argument to enable clahe
    parser.add_argument('--clahe', action='store_true', help='If clahe should be used')
    # argument to compress the latent space
    parser.add_argument('--compress', action='store_true', help='If the latent space should be compressed')

    args = parser.parse_args()

    # train the autoenc moodel
    # this requires V100s.
    gpus = args.gpus
    conf = ultrasound_autoenc(on_cluster=args.cluster, compress_latent=args.compress)
    if args.clahe:
        conf.use_clahe = True
        conf.name = f'{conf.name}_clahe'
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = args.gpus
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # # train the latent DPM
    # # NOTE: only need a single gpu
    # gpus = args.gpus
    # conf = ultrasound_autoenc_latent()
    # train(conf, gpus=gpus)
    #
    # # unconditional sampling score
    # # NOTE: a lot of gpus can speed up this process
    # gpus = args.gpus
    # conf.eval_programs = ['fid(10,10)']
    # train(conf, gpus=gpus, mode='eval')