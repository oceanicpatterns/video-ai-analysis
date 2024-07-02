import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class VideoAnalyzer:
    def __init__(self):
        self.detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.classification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.detection_model.eval()
        self.classification_model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def analyze_frame(self, frame):
        # Object detection
        img_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
        detections = self.detection_model(img_tensor)[0]

        # Scene classification
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0)
        scene_output = self.classification_model(input_tensor)
        _, predicted_scene = scene_output.max(1)

        return detections, predicted_scene.item()

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        results = {'objects': {}, 'scenes': {}}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % fps != 0:  # Process one frame per second
                continue
            
            detections, scene = self.analyze_frame(frame)
            
            for label, score in zip(detections['labels'], detections['scores']):
                if score > 0.5:  # Confidence threshold
                    label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
                    results['objects'][label_name] = results['objects'].get(label_name, 0) + 1
            
            results['scenes'][scene] = results['scenes'].get(scene, 0) + 1
        
        cap.release()
        return results

def summarize_results(results):
    summary = "Video Analysis Summary:\n"
    summary += "\nObjects detected:\n"
    for obj, count in sorted(results['objects'].items(), key=lambda x: x[1], reverse=True):
        summary += f"{obj}: {count} detections\n"
    
    summary += "\nScenes classified:\n"
    for scene, count in sorted(results['scenes'].items(), key=lambda x: x[1], reverse=True):
        summary += f"Scene {scene}: {count} frames\n"
    
    return summary

def main():
    analyzer = VideoAnalyzer()
    
    video_dir = 'videos/'
    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, video_file)
            print(f"Analyzing {video_file}...")
            try:
                results = analyzer.analyze_video(video_path)
                summary = summarize_results(results)
                print(summary)
            except Exception as e:
                print(f"Error analyzing {video_file}: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()
