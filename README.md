# Image-caption-with-LLM-and-Langchain

# Description

In this project, I'll implement the image analysis based on the ChatGPT API and Langchain. User can import an image and ask questions about this image, e.g. what object in this image, even identify the bounding box of the object.

The whole process has been devided into three parts:

1. Load image
2. Design two tools:
    * Image caption. Use this tool to describe the content of this image by the model of "Salesforce/blip-image-captioning-large". It will return a short caption describing the image.
    * Object detector. Use this tool detect objects by the model of "DetrForObjectDetection". It will return a list of all detected objects. Each element in the list in the format:"[x1, y1, x2, y2] class_name confidence_score."
3. Implement `Langchain.agent` to call these two tools and get response from OPENAI API.

**Note:** The files "functions.py" and "image_caption_Q&A.ipynb" are for test, won't affect on this project.

Reference: [Youtube Video](https://www.youtube.com/watch?v=71EOM5__vkI)
    