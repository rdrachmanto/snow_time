from PIL import Image

# Load the image
image_path = '/home/myid/dc41937/snow_time/CSWV_S6/mask_cloud/0.tif'
with Image.open(image_path) as img:
    width, height = img.size
    print(f"Resolution: {width} x {height} pixels")