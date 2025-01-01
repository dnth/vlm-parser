import re

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import spacy
import xinfer


def load_model():
    model = xinfer.create_model(
        "microsoft/Florence-2-large-ft", device="cuda", dtype="float16"
    )

    return model


def plot_zsd(image_path, output_string):
    """
    Plot bounding boxes and labels from xinfer output string

    Args:
        image_path (str): Path to the image
        output_string (str): Output string in format "class<loc_x1><loc_y1><loc_x2><loc_y2>"
    """
    # Read and convert image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Create a copy for drawing
    image_zsd = image.copy()

    # Regular expression to extract class names and coordinates
    pattern = r"(\w+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    matches = re.finditer(pattern, output_string)

    # Generate random colors for each detection
    colors = plt.cm.rainbow(np.linspace(0, 1, len(re.findall(pattern, output_string))))

    # Plot each detection
    for idx, match in enumerate(matches):
        class_name = match.group(1)
        # Convert normalized coordinates (0-1000) to image coordinates
        x1 = int(int(match.group(2)) / 1000 * width)
        y1 = int(int(match.group(3)) / 1000 * height)
        x2 = int(int(match.group(4)) / 1000 * width)
        y2 = int(int(match.group(5)) / 1000 * height)

        # Get color for this detection (convert from RGBA to RGB*255)
        color = tuple(int(c * 255) for c in colors[idx][:3])

        # Draw rectangle
        cv2.rectangle(image_zsd, (x1, y1), (x2, y2), color, 2)

        # Add label
        cv2.putText(
            image_zsd,
            class_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    return image_zsd


def get_rainbow_colors(num_detections):
    """Helper function to generate rainbow colors"""
    colors = plt.cm.rainbow(np.linspace(0, 1, num_detections))
    return [tuple(int(c * 255) for c in color[:3]) for color in colors]


def process_image(image_path):
    model = load_model()
    detailed_caption = model.infer(image_path, text="<MORE_DETAILED_CAPTION>").text

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(detailed_caption)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # Get bounding boxes
    od_boxes = model.infer(image_path, text="<OD>").text

    # Dense region caption
    drc = model.infer(image_path, text="<DENSE_REGION_CAPTION>").text

    # Zero shot detection
    # Join nouns with dots for zero shot detection
    nouns_string = ". ".join(nouns)
    zsd = model.infer(
        image_path, text=f"<CAPTION_TO_PHRASE_GROUNDING>{nouns_string}"
    ).text

    # Load and process the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio

    # copy to a new image
    image_od = image.copy()
    image_drc = image.copy()

    # Generate colors for each detection type
    od_colors = get_rainbow_colors(len(od_boxes["bboxes"]))
    drc_colors = get_rainbow_colors(len(drc["bboxes"]))

    # Draw bounding boxes and labels for object detection
    for idx, (bbox, label) in enumerate(zip(od_boxes["bboxes"], od_boxes["labels"])):
        x1, y1, x2, y2 = map(int, bbox)
        color = od_colors[idx]
        cv2.rectangle(image_od, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_od,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    # Draw bounding boxes and labels for dense region caption
    for idx, (bbox, label) in enumerate(zip(drc["bboxes"], drc["labels"])):
        x1, y1, x2, y2 = map(int, bbox)
        color = drc_colors[idx]
        cv2.rectangle(image_drc, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_drc,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    # Plot zero shot detection
    image_zsd = plot_zsd(image_path, zsd)

    extracted_info = f"Caption: {detailed_caption}\n\nNouns: {nouns}, \n\nBounding Boxes: {od_boxes}, \n\nDense Region Caption: {drc}, \n\nZero Shot Detection: {zsd}"

    return image_od, image_drc, image_zsd, extracted_info


# Update the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Image(label="Object Detection"),
        gr.Image(label="Dense Region Caption"),
        gr.Image(label="Zero Shot Detection"),
        gr.Textbox(label="Extracted Information"),
    ],
    title="Extract visual information from image",
    description="Upload an image",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
