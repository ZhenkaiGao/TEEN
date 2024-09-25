import os
from PIL import Image
from pycocotools.coco import COCO

# Define dataset paths
train_img_dir = 'coco/train2017'
val_img_dir = 'coco/val2017'
annotations_dir = 'coco/annotations'
output_dir = 'data/coco'

# Initialize COCO api
train_annotations_file = os.path.join(annotations_dir, 'instances_train2017.json')
val_annotations_file = os.path.join(annotations_dir, 'instances_val2017.json')

train_coco = COCO(train_annotations_file)
val_coco = COCO(val_annotations_file)

# Get all categories
categories = train_coco.loadCats(train_coco.getCatIds())

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to crop and save images
def crop_and_save_images(coco, img_ids, img_dir, category_name, category_output_dir):
    count = 1
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_file_path = os.path.join(img_dir, img_info['file_name'])

        # Open the image
        img = Image.open(img_file_path)
        img_width, img_height = img.size  # Get image dimensions

        # Get all annotations for the category in this image
        cat_id = coco.getCatIds(catNms=[category_name])[0]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)

        # Iterate over each annotation and crop
        for ann in anns:
            bbox = ann['bbox']
            x, y, width, height = bbox

            # Ensure the cropping area is within the image bounds
            x_end = x + width
            y_end = y + height
            if width > 0 and height > 0 and x_end <= img_width and y_end <= img_height:
                # Crop the image
                cropped_img = img.crop((x, y, x_end, y_end))

                # Ensure the cropped image is valid
                if cropped_img.size[0] > 0 and cropped_img.size[1] > 0:
                    # Create output file name (category_name_count.jpg)
                    cropped_img_name = f"{category_name}_{count}.jpg"
                    cropped_img_path = os.path.join(category_output_dir, cropped_img_name)

                    # Save the cropped image
                    cropped_img.save(cropped_img_path)
                    count += 1
                else:
                    print(f"Skipped invalid crop: {img_info['file_name']}, bbox: {bbox}")
            else:
                print(f"Skipped out-of-bounds crop: {img_info['file_name']}, bbox: {bbox}")


# Crop images for each category and store them in train and val directories
for category in categories:
    category_name = category['name']
    category_id = category['id']

    # Get all image IDs for the category
    train_image_ids = train_coco.getImgIds(catIds=[category_id])
    val_image_ids = val_coco.getImgIds(catIds=[category_id])

    # Create output directories for the category (train and val)
    train_category_output_dir = os.path.join(output_dir, 'train', category_name)
    val_category_output_dir = os.path.join(output_dir, 'val', category_name)

    if not os.path.exists(train_category_output_dir):
        os.makedirs(train_category_output_dir)
    if not os.path.exists(val_category_output_dir):
        os.makedirs(val_category_output_dir)

    # Crop images from train2017 and save them to train/category folder
    crop_and_save_images(train_coco, train_image_ids, train_img_dir, category_name, train_category_output_dir)

    # Crop images from val2017 and save them to val/category folder
    crop_and_save_images(val_coco, val_image_ids, val_img_dir, category_name, val_category_output_dir)

    # Output category progress
    print(f"Completed cropping and saving images for category: {category_name}")

print("All categories processed. Cropped images have been saved to their respective folders.")
