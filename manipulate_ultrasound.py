import argparse

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

import wandb

# read arguments from the command line and parse them
parser = argparse.ArgumentParser(description='Interpolate between two ultrasound images')
parser.add_argument('--clahe', action='store_true', help='If clahe should be used')
# add compressed argument
parser.add_argument('--compress', action='store_true', help='If the latent space should be compressed')

args = parser.parse_args()

# Initialize wandb

w = wandb.init(project="ultrasound_manipulation")

device = 'cuda:1'
conf = ultrasound_autoenc(use_clahe=args.clahe, on_cluster=False, compress_latent=args.compress)
print(conf.name)
model = LitModel(conf)
state = torch.load(f'/home/farid/Desktop/diffae_checkpoint/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

cls_conf = ultrasound_autoenc_cls(False, use_clahe=args.clahe, compress_latent=args.compress)
cls_model = ClsModel(cls_conf)
state = torch.load(f'/home/farid/Desktop/diffae_checkpoint/{cls_conf.name}/last.ckpt', map_location='cpu')
print('latent step:', state['global_step'], state['epoch'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)


# data = UltrasoundDb('datasets/interpolation/', image_size=conf.img_size, do_augment=False)  # it has only two images

# phantom data
# real_to_fake determines if we want to manipulate real images to fake images or vice versa
# uncomment the line below to manipulate real images to fake images or vice versa

# data = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/phantom_real_data/', image_size=conf.img_size, do_augment=False, use_clahe=conf.use_clahe)
# real_to_fake = True

data = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/simulated/2d_images_new/', image_size=conf.img_size, do_augment=False)
real_to_fake = False

w.name = f'{conf.name}_real_to_fake' if real_to_fake else f'{conf.name}_fake_to_real'

# read an image to manipulate
for i in range(0, len(data), 5):
    index = i
    batch = data[index]['img'][None]

    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ori = (batch + 1) / 2
    # ax[0].imshow(ori[0].permute(1, 2, 0).cpu(), cmap='gray')
    # ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
    #
    # plt.show()

    print("classifier output before manipulation:", cls_model.classifier(cond))

    direction = -1.0 if real_to_fake else 1.0

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 1, figsize=(5, 20))  # 5 rows, 2 columns

    ori = (batch + 1) / 2
    axs[0].imshow(ori[0].permute(1, 2, 0).cpu(), cmap='gray')
    axs[0].set_title('original')
    axs[0].axis('off')
    wandb_images = [wandb.Image(ori.cpu(), caption='original')]

    for i, (coeff) in enumerate([0.1, 0.25, 0.5, 0.75]):

        print("coeff:", coeff)
        cond2 = cls_model.normalize(cond)
        cond2 = cond2 + direction * coeff * math.sqrt(512) * F.normalize(cls_model.ema_classifier.weight[0][None, :], dim=1)
        cond2 = cls_model.denormalize(cond2)

        print("classifier output after manipulation:", cls_model.ema_classifier(cond2))

        # torch.manual_seed(1)
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img = model.render(xT, cond2, T=100)

        # Plot the images in the respective subplots

        axs[i+1].imshow(img[0].permute(1, 2, 0).cpu(), cmap='gray')
        axs[i+1].set_title(f'coeff: {coeff}')
        axs[i+1].axis('off')
        wandb_images.append(wandb.Image(img.cpu(), caption=f'coeff: {coeff}'))

    if real_to_fake:
        plt.savefig(f'imgs_manipulated/real_to_fake_{index}.png')
        wandb.log({f"manipulated_images/real_to_fake_{index}": wandb_images})
    else:
        plt.savefig(f'imgs_manipulated/fake_to_real_{index}.png')
        wandb.log({f"manipulated_images/fake_to_real_{index}": wandb_images})
    # plt.show()
    plt.close()