import mediapipe as mp

mp_pose = mp.solutions.pose

def classify_stroke(landmarks):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    if nose.y > left_shoulder.y and nose.y > right_shoulder.y:
        return "Backstroke"
    elif left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y:
        return "Butterfly"
    elif left_elbow.y < left_shoulder.y or right_elbow.y < right_shoulder.y:
        return "Freestyle"
    else:
        return "Breaststroke"
