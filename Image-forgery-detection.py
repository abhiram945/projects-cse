import cv2
import numpy as np
import gradio as gr

def detectForgery(pre_image_1, pre_image_2, pre_image_3, test_image):
    # Load the pre images and the image to be tested
    pre_images = [pre_image_1, pre_image_2, pre_image_3]
    test_image = test_image

    # Resize all images to fit in ratio
    max_size = 800

    # Pre images
    for i in range(len(pre_images)):
        height, width, _ = pre_images[i].shape
        if height > width:
            ratio = max_size / height
        else:
            ratio = max_size / width
        pre_images[i] = cv2.resize(pre_images[i], (int(width * ratio), int(height * ratio)))

    # Test image
    height, width, _ = test_image.shape
    if height > width:
        ratio = max_size / height
    else:
        ratio = max_size / width
    test_image = cv2.resize(test_image, (int(width * ratio), int(height * ratio)))

    # Compute the difference between the pre images and the test image
    diffs = [cv2.absdiff(img, test_image) for img in pre_images]

    # Convert the difference images to grayscale and threshold them
    thresh = 150
    diffs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in diffs]
    diffs = [cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1] for img in diffs]

    # Find the contours of the differences in each image
    contours = [cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for img in diffs]

    # Draw the contours on the test image
    for i in range(len(contours)):
        for contour in contours[i]:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # If there are no differences, mark the test image as authentic
    if all(len(contours[i]) == 0 for i in range(len(contours))):
        return "The test image is authentic", np.zeros_like(test_image)
    else:
        return "The test image is a forgery", test_image

# Create the Gradio interface
inputs = [
    gr.inputs.Image(label="Pre-image 1"),
    gr.inputs.Image(label="Pre-image 2"),
    gr.inputs.Image(label="Pre-image 3"),
    gr.inputs.Image(label="Test Image")
]

output=["text", "image"]
interface = gr.Interface(fn=detectForgery, inputs=inputs, outputs=output, title="Forgery Detection", 
                         description="Upload three pre-images and a test image to check if the test image is a forgery.")

interface.launch()
