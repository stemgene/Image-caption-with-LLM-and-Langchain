from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

def get_image_caption(image_path):
    """
    generate short captions for the provided image
    
    Args:
        image_path (str): The path to the image file
    
    Returns:
        str: A string representing the caption for the image
    
    """
    image = Image.open(image_path).convert("RGB")

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else 'cpu' # cuda
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption



def detect_objects(image_path):
    # Detect the objects in the given image
    image = Image.open(image_path).convert("RGB")
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


if __name__ == '__main__':
    image_path = r"C:\Users\hydon\Downloads\horseback_riding.jpg"
    caption = get_image_caption(image_path)
    print(caption)
    detections = detect_objects(image_path)
    print(detections)
