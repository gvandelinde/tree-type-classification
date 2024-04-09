#transform and generate images for low sample datasets (after importing blurred_trees)
import os
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms
import random

# Function to identify categories with lower samples
def identify_low_sample_categories(images_folder):
    image_counts = {}
    for cat_folder in os.listdir(images_folder):
        cat_path = os.path.join(images_folder, cat_folder)
        image_counts[cat_folder] = len(os.listdir(cat_path))
    max_samples = max(image_counts.values())
    low_sample_categories = [label for label, count in image_counts.items() if count < max_samples]
    return low_sample_categories

# Define data augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))  # Random translation
])

# Function to apply transformations and save additional images
def augment_and_save_images(images_folder, low_sample_categories, transforms):

    image_counts = {}
    for cat_folder in os.listdir(images_folder):
        cat_path = os.path.join(images_folder, cat_folder)
        image_counts[cat_folder] = len(os.listdir(cat_path))
        
    cat_counts = Counter(image_counts)
    max_samples = max(cat_counts.values())
    for category in low_sample_categories:
        # os.makedirs('/kaggle/working/'+"tree"+"/"+category)
        cat_folder = os.path.join(images_folder, category)
        num_samples = cat_counts[category]

        # Calculate how many additional images to generate
        num_additional_images = max_samples - num_samples
        for i in range (num_additional_images):
            while True:
                random_index = random.randint(0, len(os.listdir(cat_folder)) - 1)
                randomfilename=os.listdir(cat_folder)[random_index]
                if "augmented" not in randomfilename:
                    break
            image_path = os.path.join(cat_folder, randomfilename)
            image = Image.open(image_path)

            # Apply transformations and save additional images
            transformed_image = transforms(image)
            new_filename = f"augmented_{num_samples + i + 1}_{randomfilename}"  # Append prefix to filename
            new_image_path = os.path.join(images_folder, new_filename)
            transformed_image.save(new_image_path)
# Transform and generate new blurred images for categories with smaller samples
images_folder = 'Model/blurred_trees'
low_sample_categories = identify_low_sample_categories(images_folder)
augment_and_save_images(images_folder, low_sample_categories, augmentation_transforms)
