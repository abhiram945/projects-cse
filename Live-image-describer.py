import cv2
import io
import numpy as np
import pyttsx3
import tempfile
import streamlit as st
from transformers import pipeline #Stable Diffusion Pipeline algo
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from PIL import Image

# Load YOLOv3 model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to perform object detection using YOLOv3
def perform_object_detection(frame):
    height, width = frame.shape[:2]

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    detected_objects = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                x, y, w, h = obj[0:4] * np.array([width, height, width, height])
                x = int(x - w/2)
                y = int(y - h/2)
                w = int(w)
                h = int(h)

                label = f"{classes[class_id]}: {confidence:.2f}"
                detected_objects.append(label)

    return detected_objects

# Function to generate a story based on the scene description
def generate_story(scene):
    template = '''You are a story teller.
                You can generate a short story based on a simple
                narrative, the story should be no more than 30 words:

                CONTEXT:{scene}
                STORY:'''

    prompt = PromptTemplate(
        input_variables=["scene"],
        template=template
    )

    chain = LLMChain(llm=OpenAI(temperature=1, openai_api_key="OPEN_AI_API_KEY_HERE"), prompt=prompt)

    story = chain.run(scene)
    return story

# Function to speak the provided text
def Speak(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[0].id)
    engine.setProperty('rate', 150)
    engine.say(Text)
    engine.runAndWait()

# Function to convert image to text using Hugging Face pipeline
def img2text(img):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
    text = image_to_text(img)
    return text[0]["generated_text"]

# Function to convert image bytes to a PIL image object
def img2pil(img):
    img_pil = Image.open(io.BytesIO(img))
    return img_pil

# Main function to create Streamlit web application
def main():
    st.header("Turn Images into Audio Stories")

    uploaded_file = st.file_uploader("Choose an image..", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(bytes_data)
            file_path = file.name

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        scene = img2text(file_path)
        story = generate_story(scene)

        with st.expander("Scene"):
            st.write(scene)

        with st.expander("Story"):
            st.write(story)

        speak = st.button("Speak")
        if speak:
            Speak(f"Scene: {scene}")
            Speak(f"Story: {story}")

        # Perform object detection and display detected objects
        img_cv2 = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        detected_objects = perform_object_detection(img_cv2)

        st.write("Detected Objects:")
        for obj in detected_objects:
            st.write(obj)
    else:
        img = st.camera_input(label="Take a Photo of the Scene")
        if img is not None:
            st.image(img, caption="Taken Photo", use_column_width=True)
            bytes_data = img.getvalue()
            with tempfile.NamedTemporaryFile(delete=False) as file:
                file.write(bytes_data)
                file_path = file.name

            scene = img2text(file_path)
            story = generate_story(scene)

            with st.expander("Scene"):
                st.write(scene)

            with st.expander("Story"):
                st.write(story)

            speak = st.button("Speak")
            if speak:
                Speak(f"Scene: {scene}")
                Speak(f"Story: {story}")

            # Perform object detection and display detected objects
            img_cv2 = cv2.cvtColor(np.array(Image.open(io.BytesIO(img.getvalue()))), cv2.COLOR_RGB2BGR)
            detected_objects = perform_object_detection(img_cv2)

            st.write("Detected Objects:")
            for obj in detected_objects:
                st.write(obj)

if __name__ == "__main__":
    main()



