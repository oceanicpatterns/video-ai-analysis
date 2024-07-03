import cv2
import mediapipe as mp
from stroke_classifier import classify_stroke
from technique_analyzer import analyze_technique
from feedback_generator import prioritize_feedback

mp_pose = mp.solutions.pose

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    strokes = []
    technique_feedback = []
    positive_feedback = []
    equipment = []
    frame_count = 0

    while cap.isOpened() and frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            stroke = classify_stroke(results.pose_landmarks)
            strokes.append(stroke)
            improvements, positives = analyze_technique(results.pose_landmarks, stroke)
            technique_feedback.extend(improvements)
            positive_feedback.extend(positives)
            # Equipment detection logic can be added here

        frame_count += 1

    cap.release()

    most_common_stroke = max(set(strokes), key=strokes.count) if strokes else "Unknown"
    prioritized_improvements, prioritized_positive_feedback = prioritize_feedback(technique_feedback, positive_feedback)

    return most_common_stroke, prioritized_improvements, prioritized_positive_feedback, equipment
