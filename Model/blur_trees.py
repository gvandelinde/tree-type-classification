from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw, ImageFilter
import os

# Configuration for the inference client with a set confidence threshold
CUSTOM_CONFIGURATION = InferenceConfiguration(confidence_threshold=0.1)

# Define the types of trees that will be classified
TREE_TYPES = ['Dood', 'Goed', 'Matig', 'Redelijk', 'Slecht', 'Zeer Slecht']

# Directories for input and output images
INPUT_PATH = "RetrieveGSV/images"
OUTPUT_PATH = "Model/blurred_trees"

# Initialize the inference client with API details
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="kiW8upICXaWRP7kAm4Ct"
)

def polygon_area(points):
    """
    Calculates the area of a polygon given its vertices.
    
    :param points: A list of (x, y) tuples representing the vertices of the polygon.
    :return: The area of the polygon.
    """
    area = 0  # Initialize the area
    n = len(points)  # Number of points

    # Calculate the sum of the cross-products
    for i in range(n):
        j = (i + 1) % n  # Next vertex index, wrapping around using modulo
        area += points[i][0] * points[j][1] - points[i][1] * points[j][0]

    # Final area calculation
    area = abs(area) / 2.0
    return area

def blur_around_tree(tree_image, json_outline):
    """
    Blurs the areas around detected trees in an image based on outlines provided in JSON format.

    :param tree_image: A PIL.Image object of the original image.
    :param json_outline: A dictionary containing the outlines of detected objects.
    :return: A tuple (has_tree, tree_area, final_image) where:
             - has_tree is a boolean indicating if a tree was detected,
             - tree_area is the fraction of the image area covered by trees,
             - final_image is the resulting PIL.Image object with areas around trees blurred.
    """
    blurred_image = tree_image.filter(ImageFilter.GaussianBlur(15))  # Apply Gaussian blur
    mask = Image.new('L', tree_image.size, 0)  # Create a blank mask for drawing polygons
    draw = ImageDraw.Draw(mask)

    has_tree = False  # Flag to indicate if any tree was detected
    tree_area = 0  # Initialize total area covered by trees

    for obj in json_outline['predictions']:
        if obj['class'] == 'tree':
            has_tree = True
            polygon = [(point['x'], point['y']) for point in obj['points']]  # Convert points to PIL-compatible format
            area = polygon_area(polygon)  # Calculate polygon area
            tree_area += area / (tree_image.size[0] * tree_image.size[1])  # Add to total tree area
            
            draw.polygon(polygon, fill=255)  # Draw polygon on mask

    final_image = Image.composite(tree_image, blurred_image, mask)  # Apply mask to combine images
    return has_tree, tree_area, final_image

def del_file(tree_path):
    """
    Deletes a file if it exists.

    :param tree_path: The path to the file to be deleted.
    """
    if os.path.exists(tree_path):
        os.remove(tree_path)

# Process each tree type
for tree_type in TREE_TYPES:
    output_tree_type_path = os.path.join(OUTPUT_PATH, tree_type)  # Construct output path
    if not os.path.exists(output_tree_type_path):
        os.makedirs(output_tree_type_path)  # Create directory if it doesn't exist

for tree_type in TREE_TYPES:
    input_tree_type_path = os.path.join(INPUT_PATH, tree_type)  # Construct input path
    output_tree_type_path = os.path.join(OUTPUT_PATH, tree_type)  # Construct output path
    for filename in os.listdir(input_tree_type_path):
        input_tree_path = os.path.join(input_tree_type_path, filename)  # Full path to input image
        output_tree_path = os.path.join(output_tree_type_path, filename)  # Full path to output image

        # Perform inference to get tree outlines
        with CLIENT.use_configuration(CUSTOM_CONFIGURATION):
            json_outline = CLIENT.infer(input_tree_path, model_id="cameraexplorer/1")
        
        original_image = Image.open(input_tree_path)  # Open the input image
        has_tree, tree_area, blurred_image = blur_around_tree(original_image, json_outline)  # Apply blurring

        if has_tree:
            if tree_area > 0.05:
                blurred_image.save(output_tree_path)
            
                