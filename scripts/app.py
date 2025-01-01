import cv2
import gradio as gr
import numpy as np
import spacy
import xinfer


def load_model():
    model = xinfer.create_model(
        "microsoft/Florence-2-large-ft", device="cuda", dtype="float16"
    )

    return model


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

    # Load and process the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio

    # copy to a new image
    image_od = image.copy()
    image_drc = image.copy()

    # Draw bounding boxes and labels for object detection
    for bbox, label in zip(od_boxes["bboxes"], od_boxes["labels"]):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_od, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image_od,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    # Draw bounding boxes and labels for dense region caption
    for bbox, label in zip(drc["bboxes"], drc["labels"]):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_drc, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image_drc,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    extracted_info = f"Caption: {detailed_caption}\n\nNouns: {nouns}, \n\nBounding Boxes: {od_boxes}, \n\nDense Region Caption: {drc}"

    return image_od, image_drc, extracted_info


# Update the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Image(label="Object Detection"),
        gr.Image(label="Dense Region Caption"),
        gr.Textbox(label="Extracted Information"),
    ],
    title="Extract information from image",
    description="Upload an image and get extracted information with detected objects",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
