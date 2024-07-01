import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import torch
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize
from pytorchvideo.data.encoded_video import EncodedVideo

def load_model():
    model = r3d_18(pretrained=True)
    model.eval()
    return model

def preprocess_video(video_path):
    clip = VideoFileClip(video_path)
    frames = []
    for frame in clip.iter_frames(fps=16):
        frame = cv2.resize(frame, (112, 112))
        frame = frame / 255.0
        frames.append(frame)
    frames = np.array(frames)
    frames = torch.from_numpy(frames.transpose(3, 0, 1, 2)).float()
    return frames.unsqueeze(0)

def analyze_video(video_path):
    model = load_model()
    frames = preprocess_video(video_path)
    
    with torch.no_grad():
        output = model(frames)
    
    # Get the top 5 predicted classes
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    
    # Load class labels (you need to provide a file with class labels)
    with open('kinetics_classnames.txt', 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    
    top_categories = [categories[idx] for idx in pred[0]]
    
    return top_categories

def main():
    video_path = input("Enter the path to your video file: ")
    results = analyze_video(video_path)
    
    print("Video Analysis Results:")
    print("This video is about:")
    for i, category in enumerate(results, 1):
        print(f"{i}. {category}")

if __name__ == "__main__":
    main()
