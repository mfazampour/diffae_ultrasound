from templates import *


# def ffhq128_autoenc_cls():
#     conf = ffhq128_autoenc_130M()
#     conf.train_mode = TrainMode.manipulate
#     conf.manipulate_mode = ManipulateMode.celebahq_all
#     conf.manipulate_znormalize = True
#     conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
#     conf.batch_size = 32
#     conf.lr = 1e-3
#     conf.total_samples = 300_000
#     # use the pretraining trick instead of contiuning trick
#     conf.pretrain = PretrainConfig(
#         '130M',
#         f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
#     )
#     conf.name = 'ffhq128_autoenc_cls'
#     return conf


def ultrasound_autoenc_cls(on_cluster, use_clahe=False, compress_latent=False):
    conf = ultrasound_autoenc(on_cluster=on_cluster, use_clahe=use_clahe, compress_latent=compress_latent)

    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.ultrasound_sim_real  # what is this?
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'{conf.base_dir}/{ultrasound_autoenc(use_clahe=use_clahe, on_cluster=on_cluster, compress_latent=compress_latent).name}/latent.pkl'
    conf.batch_size = 48
    conf.lr = 1e-4
    conf.num_epochs = 35
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'{conf.base_dir}/{ultrasound_autoenc(use_clahe=use_clahe, on_cluster=on_cluster, compress_latent=compress_latent).name}/last.ckpt',
    )
    conf.name = 'ultrasound_autoenc_cls'
    if use_clahe:
        conf.name = f'ultrasound_autoenc_cls_clahe'
    if compress_latent:
        conf.name += '_compressed'
    return conf


# def ffhq256_autoenc_cls():
#     '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
#     conf = ffhq256_autoenc()
#     conf.train_mode = TrainMode.manipulate
#     conf.manipulate_mode = ManipulateMode.celebahq_all
#     conf.manipulate_znormalize = True
#     conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
#     conf.batch_size = 32
#     conf.lr = 1e-3
#     conf.total_samples = 300_000
#     # use the pretraining trick instead of contiuning trick
#     conf.pretrain = PretrainConfig(
#         '130M',
#         f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
#     )
#     conf.name = 'ffhq256_autoenc_cls'
#     return conf
