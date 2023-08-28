#! matplotlib inline
import argparse

from templates import *
from templates_latent import *
import numpy as np

import matplotlib.pyplot as plt

# read arguments from the command line and parse them
parser = argparse.ArgumentParser(description='Interpolate between two ultrasound images')
parser.add_argument('--clahe', action='store_true', help='If clahe should be used')
# add compressed argument
parser.add_argument('--compress', action='store_true', help='If the latent space should be compressed')

args = parser.parse_args()


device = 'cuda:1'
conf = ultrasound_autoenc(use_clahe=args.clahe, on_cluster=False, compress_latent=args.compress)
print(conf.name)
model = LitModel(conf)
state = torch.load(f'/home/farid/Desktop/diffae_checkpoint/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

dataset = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/simulated/2d_images_new/', image_size=conf.img_size, do_augment=False, only_load_synthetic=True, use_clahe=conf.use_clahe)
data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)

query_set = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/phantom_real_data/', image_size=conf.img_size, do_augment=False, use_clahe=conf.use_clahe)
# read the two images and stack them into a batch
query_img = query_set[50]['img']


for i in range(0, len(dataset), 20):
    # read the two images and stack them into a batch
    batch = torch.stack([
        query_img,
        dataset[i]['img'],
    ])

    # plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5, cmap='gray')
    # plt.show()

    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ori = (batch + 1) / 2
    # ax[0].imshow(ori[0].permute(1, 2, 0).cpu(), cmap='gray')
    # ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
    # plt.show()

    alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)
    intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

    def cos(a, b):
        a = a.view(-1)
        b = b.view(-1)
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return (a * b).sum()


    theta = torch.arccos(cos(xT[0], xT[1]))
    print(theta)
    x_shape = xT[0].shape
    intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) *
              xT[1].flatten(0, 2)[None]) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape)

    pred = model.render(intp_x, intp, T=20)

    # torch.manual_seed(1)
    fig, ax = plt.subplots(1, 10, figsize=(5 * 10, 5))
    for i in range(len(alpha)):
        ax[i].imshow(pred[i].permute(1, 2, 0).cpu(), cmap='gray')
        ax[i].axis('off')

    plt.show()
