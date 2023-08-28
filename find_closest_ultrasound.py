from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda:1'
conf = ultrasound_autoenc(on_cluster=False)
conf.batch_size = 100
print(conf.name)
model = LitModel(conf)
state = torch.load(f'/home/farid/Desktop/diffae_checkpoint/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

cls_conf = ultrasound_autoenc_cls(False)
cls_model = ClsModel(cls_conf)
state = torch.load(f'/home/farid/Desktop/diffae_checkpoint/{cls_conf.name}/last.ckpt', map_location='cpu')
print('latent step:', state['global_step'], state['epoch'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

dataset = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/simulated/2d_images_new/', image_size=conf.img_size, do_augment=False, only_load_synthetic=True)
data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)

query_set = UltrasoundDb('/mnt/polyaxon/data1/ct_us_registration_prius/phantom_data/phantom_real_data/', image_size=conf.img_size, do_augment=False)
# read the two images and stack them into a batch
batch = query_set[100]['img'][None]

query_output = model.encode(batch.to(device)).squeeze()

fixed_vector = F.normalize(cls_model.ema_classifier.weight[0][None, :], dim=1).squeeze()

# List to store images and their corresponding similarity scores
image_similarity_list = []

for batch_data in tqdm(data_loader, desc="Processing images"):
    batch_images, batch_index = batch_data['img'], batch_data['index']
    with torch.no_grad():
        batch_outputs = model.encode(batch_images.to(device))

    for i, dataset_output in enumerate(batch_outputs):
        # Calculate the distance vector
        distance_vector = query_output - dataset_output

        # Calculate the dot product between distance vector and fixed vector
        dot_product = torch.dot(distance_vector, fixed_vector)

        # Calculate the cosine of the angle
        cosine_angle = dot_product / (torch.norm(distance_vector) * torch.norm(fixed_vector))

        # Calculate the angle in degrees
        angle = torch.acos(cosine_angle) * 180 / torch.pi

        # Calculate how close the angle is to 90 degrees
        difference_from_90 = torch.abs(angle - 90)

        # Append the image and similarity score
        image_similarity_list.append((batch_images[i], difference_from_90, batch_index[i]))

# Sort the images based on similarity
sorted_images = sorted(image_similarity_list, key=lambda x: x[1], reverse=True)

# Store sorted images
sorted_images_only = [item[0] for item in sorted_images]

# Now, sorted_images_only contains the sorted images based on the similarity criteria
# You can visualize the images using matplotlib or save them to disk

# Visualize the first 10 images
fig = plt.figure(figsize=(20, 5))
# visualize the original image
ax = fig.add_subplot(1, 11, 1)
ax.imshow(batch[0].permute(1, 2, 0), cmap='gray')
ax.axis('off')
for i in range(10):
    ax = fig.add_subplot(1, 11, i + 2)
    ax.imshow(sorted_images_only[i].permute(1, 2, 0), cmap='gray')
    ax.axis('off')
plt.show()
