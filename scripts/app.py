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


def get_class_colors(class_labels):
    """Helper function to generate consistent colors for each unique class"""
    unique_classes = list(set(class_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    # Create a mapping of class names to colors
    color_map = {
        class_name: tuple(int(c * 255) for c in colors[i][:3])
        for i, class_name in enumerate(unique_classes)
    }
    return color_map


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

    # Region proposal
    rp = model.infer(image_path, text="<REGION_PROPOSAL>").text

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio

    image_od = image.copy()
    image_drc = image.copy()
    image_rp = image.copy()

    # Generate color mappings for each detection type
    od_color_map = get_class_colors(od_boxes["labels"])
    drc_color_map = get_class_colors(drc["labels"])
    rp_color_map = get_class_colors(rp["labels"])

    # Draw bounding boxes and labels for object detection
    for bbox, label in zip(od_boxes["bboxes"], od_boxes["labels"]):
        x1, y1, x2, y2 = map(int, bbox)
        color = od_color_map[label]
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
    for bbox, label in zip(drc["bboxes"], drc["labels"]):
        x1, y1, x2, y2 = map(int, bbox)
        color = drc_color_map[label]
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

    # Plot region proposal
    for bbox, label in zip(rp["bboxes"], rp["labels"]):
        x1, y1, x2, y2 = map(int, bbox)
        color = rp_color_map[label]
        cv2.rectangle(image_rp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_rp,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    return (
        image_od,
        image_drc,
        image_zsd,
        image_rp,
        detailed_caption,  # Caption
        ", ".join(nouns),  # Tags
        str(od_boxes),  # Bounding Boxes
        str(drc),  # Dense Region Caption
        zsd,  # Zero Shot Detection
        str(rp),  # Region Proposal
    )


# Update the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Image(label="Object Detection"),
        gr.Image(label="Dense Region Caption"),
        gr.Image(label="Zero Shot Detection"),
        gr.Image(label="Region Proposal"),
        gr.Textbox(label="Detailed Caption", lines=3),
        gr.Textbox(label="Tags", lines=2),
        gr.Textbox(label="Bounding Boxes", lines=4),
        gr.Textbox(label="Dense Region Caption", lines=4),
        gr.Textbox(label="Zero Shot Detection", lines=4),
        gr.Textbox(label="Region Proposal", lines=4),
    ],
    title="Extract visual information from image",
    description="Upload an image",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
