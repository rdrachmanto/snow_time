import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None

# Paths
base_dir = '/home/myid/dc41937/snow_time/CSWV_S6'
image_dir = os.path.join(base_dir, 'image')
mask_cloud_dir = os.path.join(base_dir, 'mask_cloud')
mask_snow_dir = os.path.join(base_dir, 'mask_snow')
output_dir = '/home/myid/dc41937/snow_time/data2'

# Patch size
patch_size = 512

# Colors for masks
SNOW_COLOR = (173, 216, 230)  # Light Blue for snow
CLOUD_COLOR = (255, 255, 255)  # White for cloud


def create_patches():
    # Ensure output directory exists
    os.makedirs(os.path.join(output_dir, 'all_patches'), exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            mask_cloud_path = os.path.join(mask_cloud_dir, filename)
            mask_snow_path = os.path.join(mask_snow_dir, filename)

            # Load images
            image = Image.open(image_path).convert('RGB')
            mask_cloud = Image.open(mask_cloud_path).convert('L')
            mask_snow = Image.open(mask_snow_path).convert('L')

            # Get dimensions
            width, height = image.size
            patch_num = 0

            # Iterate over the image with no overlap
            for i in range(0, width, patch_size):
                for j in range(0, height, patch_size):
                    # Define patch box
                    box = (i, j, i + patch_size, j + patch_size)

                    # Extract patches
                    image_patch = image.crop(box)
                    mask_cloud_patch = mask_cloud.crop(box)
                    mask_snow_patch = mask_snow.crop(box)

                    # Combine masks
                    combined_mask = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    mask_cloud_array = np.array(mask_cloud_patch)
                    mask_snow_array = np.array(mask_snow_patch)

                    # Apply colors to combined mask
                    combined_mask[mask_cloud_array > 0] = CLOUD_COLOR
                    combined_mask[mask_snow_array > 0] = SNOW_COLOR

                    combined_mask_img = Image.fromarray(combined_mask)

                    # Stitch image and combined mask
                    stitched = Image.new('RGB', (patch_size * 2, patch_size))
                    stitched.paste(image_patch, (0, 0))
                    stitched.paste(combined_mask_img, (patch_size, 0))

                    # Save the patch
                    patch_name = f"{os.path.splitext(filename)[0]}_{patch_num}.png"
                    stitched.save(os.path.join(output_dir, 'all_patches', patch_name))
                    patch_num += 1


# Generate patches and save them all
create_patches()

# Split patches into train, validation, and test sets
all_patches = os.listdir(os.path.join(output_dir, 'all_patches'))
train_patches, test_patches = train_test_split(all_patches, test_size=0.2, random_state=42)
train_patches, val_patches = train_test_split(train_patches, test_size=0.25, random_state=42)  # 0.25 of 80% is 20%


# Helper function to move patches to train, val, test folders
def save_split(patches, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for patch in patches:
        src = os.path.join(output_dir, 'all_patches', patch)
        dst = os.path.join(split_dir, patch)
        os.rename(src, dst)


# Save splits
save_split(train_patches, 'train')
save_split(val_patches, 'val')
save_split(test_patches, 'test')

print("Patches created and split into train, val, and test sets.")