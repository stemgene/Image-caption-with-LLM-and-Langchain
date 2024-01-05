from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import requests

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a short caption describing the image"
    
    def _run(self, image_url):
        # image = Image.open(img_path).convert("RGB")
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cuda" if torch.cuda.is_available() else 'cpu' # cuda
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption
    
    def _arun(self, query:str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."
    
    def _run(self, image_url):
  
        #image = Image.open(img_path).convert("RGB")
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        device = "cuda" if torch.cuda.is_available() else 'cpu' # cuda

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors='pt').to(device)
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            # box = [round(i, 2) for i in box.tolist()]
            # print(
            #     f"Detected {model.config.id2label[label.item()]} with confidence "
            #     f"{round(score.item(), 3)} at location {box}"
            # )
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))
        
        return detections
    
    def _arun(self, query:str):
        pass