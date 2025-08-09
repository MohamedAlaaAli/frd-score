import os
from PIL import Image

def resize_images_in_dir(root_dir, size=(224, 224), extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Recursively resize all images in a directory and its subdirectories to the given size.
    Overwrites the images in place.
    """
    count = 0
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(extensions):
                img_path = os.path.join(root, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(size, Image.BILINEAR)
                    img.save(img_path)  # overwrite
                    count += 1
                except Exception as e:
                    print(f"❌ Failed to process {img_path}: {e}")
    print(f"✅ Resized {count} images to {size[0]}x{size[1]} in '{root_dir}'.")


if __name__ == "__main__":
    dataset_dir = "frd/frd_v1/datasets/BUSI"  # <-- change this to your dataset folder
    resize_images_in_dir(dataset_dir)
