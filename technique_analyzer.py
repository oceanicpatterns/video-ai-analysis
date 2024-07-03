from utils import calculate_angle, BODY_ALIGNMENT_THRESHOLD, LEG_MOVEMENT_THRESHOLD, ARM_MOVEMENT_THRESHOLD, SHOULDER_WIDTH_THRESHOLD
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def analyze_technique(landmarks, stroke):
    if stroke == "Freestyle":
        return analyze_freestyle(landmarks)
    elif stroke == "Backstroke":
        return analyze_backstroke(landmarks)
    elif stroke == "Breaststroke":
        return analyze_breaststroke(landmarks)
    elif stroke == "Butterfly":
        return analyze_butterfly(landmarks)
    else:
        return [], []

def analyze_freestyle(landmarks):
    feedback = []
    positive_feedback = []

    # Extract relevant landmarks
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Head position analysis
    if nose.y < (left_shoulder.y + right_shoulder.y) / 2:
        feedback.append("Keep your head down: Look at the bottom of the pool to maintain proper body alignment")
    else:
        positive_feedback.append("Good head position: Your eyes are looking down, helping to keep your hips high")

    # Body alignment analysis
    body_line = np.polyfit([left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x],
                           [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y], 1)
    body_angle = np.arctan(body_line[0]) * 180 / np.pi
    if abs(body_angle) > 10:
        feedback.append(f"Improve body alignment: Your body is at a {abs(body_angle):.1f} degree angle. Aim for a more horizontal position")
    else:
        positive_feedback.append("Good body alignment: Your body is maintaining a horizontal position in the water")

    # Hip position analysis
    hip_position = (left_hip.y + right_hip.y) / 2
    shoulder_position = (left_shoulder.y + right_shoulder.y) / 2
    if hip_position - shoulder_position > 0.1:
        feedback.append("Lift your hips: Keep your hips high in the water to reduce drag")
    else:
        positive_feedback.append("Good hip position: Your hips are high, reducing drag and improving efficiency")

    # Arm extension analysis
    left_arm_extension = np.linalg.norm(np.array([left_wrist.x - left_shoulder.x, left_wrist.y - left_shoulder.y]))
    right_arm_extension = np.linalg.norm(np.array([right_wrist.x - right_shoulder.x, right_wrist.y - right_shoulder.y]))
    if left_arm_extension < 0.4 or right_arm_extension < 0.4:
        feedback.append("Extend your arms further: Reach forward more on each stroke to maximize your distance per stroke")
    else:
        positive_feedback.append("Good arm extension: You're reaching forward well, maximizing your distance per stroke")

    # High elbow catch analysis
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if left_elbow_angle > 120 or right_elbow_angle > 120:
        feedback.append("Improve your catch: Keep your elbow high during the pull phase for better propulsion")
    else:
        positive_feedback.append("Good high elbow catch: Your elbow position during the pull phase is effective")

    # Body rotation analysis
    shoulder_rotation = abs(left_shoulder.z - right_shoulder.z)
    if shoulder_rotation < 0.1:
        feedback.append("Increase body rotation: Rotate your body more with each stroke for better efficiency")
    else:
        positive_feedback.append("Good body rotation: You're rotating well with each stroke, which helps with efficiency")

    # Kick analysis
    kick_amplitude = abs(left_ankle.y - right_ankle.y)
    if kick_amplitude > 0.3:
        feedback.append("Refine your kick: Keep your kicks narrow and rhythmic to maintain streamlined body position")
    else:
        positive_feedback.append("Good kick technique: Your kicks are compact and efficient")

    # Hand entry analysis
    if abs(left_wrist.x - right_wrist.x) > 0.3:
        feedback.append("Narrow your hand entry: Your hands should enter the water at shoulder width")
    else:
        positive_feedback.append("Good hand entry: Your hands are entering the water at an appropriate width")

    return feedback, positive_feedback


def analyze_breaststroke(landmarks):
    feedback = []
    positive_feedback = []

    # Extract relevant landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Calculate shoulder width
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    # Arm pull analysis
    arm_width = abs(left_wrist.x - right_wrist.x)
    if arm_width > shoulder_width * SHOULDER_WIDTH_THRESHOLD:
        feedback.append("Narrow your arm pull: Your hands are too wide during the pull phase")
    elif arm_width < shoulder_width:
        feedback.append("Widen your arm pull: Your hands are too close during the pull phase")
    else:
        positive_feedback.append("Good arm width: Your hands are at an appropriate width during the pull phase")

    # Timing analysis
    if abs(left_elbow.y - right_elbow.y) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Improve timing: Coordinate your arm pull and leg kick for better propulsion")
    else:
        positive_feedback.append("Good timing: Your arm pull and leg kick are well-coordinated")

    # Kick analysis
    knee_width = abs(left_knee.x - right_knee.x)
    if knee_width > shoulder_width * SHOULDER_WIDTH_THRESHOLD:
        feedback.append("Narrow your kick: Keep your knees within shoulder width for a more efficient kick")
    else:
        positive_feedback.append("Good kick width: Your knees are at an appropriate width for an efficient breaststroke kick")

    # Pull analysis
    pull_width = max(abs(left_elbow.x - left_shoulder.x), abs(right_elbow.x - right_shoulder.x))
    if pull_width > ARM_MOVEMENT_THRESHOLD:
        feedback.append("Improve pull width: Keep your arm pull narrow and efficient")
    else:
        positive_feedback.append("Efficient pull: Your arm pull is appropriately narrow")

    # Body position analysis
    body_line = np.polyfit([left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x],
                           [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y], 1)
    body_angle = np.arctan(body_line[0]) * 180 / np.pi
    if abs(body_angle) > 15:
        feedback.append(f"Improve body position: Your body is at a {abs(body_angle):.1f} degree angle. Aim for a more horizontal position")
    else:
        positive_feedback.append("Good body position: You're maintaining a horizontal body position")

    # Head position analysis
    if nose.y < (left_shoulder.y + right_shoulder.y) / 2:
        feedback.append("Lower your head: Keep your head in line with your spine to reduce drag")
    else:
        positive_feedback.append("Good head position: Your head is well-aligned with your spine")

    # Glide analysis
    if abs(left_wrist.x - right_wrist.x) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Improve glide position: Keep your arms together in streamline during the glide phase")
    else:
        positive_feedback.append("Efficient glide: You're maintaining a good streamline position during the glide phase")

    # Arm recovery analysis
    if left_elbow.y < left_shoulder.y or right_elbow.y < right_shoulder.y:
        feedback.append("Improve arm recovery: Keep your elbows lower than your shoulders during recovery")
    else:
        positive_feedback.append("Good arm recovery: Your elbows are positioned correctly during the recovery phase")

    # Kick symmetry analysis
    if abs(left_ankle.y - right_ankle.y) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Improve kick symmetry: Ensure both legs are moving symmetrically")
    else:
        positive_feedback.append("Good kick symmetry: Your legs are moving symmetrically")

    # Hip position analysis
    if abs(left_hip.y - right_hip.y) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Level your hips: Keep your hips aligned to maintain a streamlined position")
    else:
        positive_feedback.append("Good hip position: Your hips are well-aligned, contributing to a streamlined position")

    return feedback, positive_feedback


def analyze_backstroke(landmarks):
    feedback = []
    positive_feedback = []

    # Extract relevant landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Head position analysis
    if nose.y < (left_shoulder.y + right_shoulder.y) / 2:
        feedback.append("Keep your head back: Look straight up to maintain proper body alignment")
    else:
        positive_feedback.append("Good head position: Your head is well-aligned, helping to keep your hips high")

    # Body position analysis
    body_line = np.polyfit([left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x],
                           [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y], 1)
    body_angle = np.arctan(body_line[0]) * 180 / np.pi
    if abs(body_angle) > 15:
        feedback.append(f"Improve body position: Your body is at a {abs(body_angle):.1f} degree angle. Aim for a more horizontal position")
    else:
        positive_feedback.append("Good body position: You're maintaining a horizontal body position")

    # Arm entry analysis
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    arm_entry_width = abs(left_wrist.x - right_wrist.x)
    if arm_entry_width > shoulder_width * 1.2:
        feedback.append("Narrow your arm entry: Your hands should enter the water at shoulder width")
    else:
        positive_feedback.append("Good arm entry: Your hands are entering the water at an appropriate width")

    # Pull analysis
    if left_elbow.y < left_shoulder.y or right_elbow.y < right_shoulder.y:
        feedback.append("Improve your pull: Keep your elbow below your shoulder during the pull phase")
    else:
        positive_feedback.append("Good pull technique: Your elbow position during the pull phase is effective")

    # Kick analysis
    kick_amplitude = abs(left_ankle.y - right_ankle.y)
    if kick_amplitude > 0.3:
        feedback.append("Reduce kick amplitude: Keep your kicks smaller and more frequent for better efficiency")
    else:
        positive_feedback.append("Good kick technique: Your kicks are compact and efficient")

    # Body rotation analysis
    shoulder_rotation = abs(left_shoulder.z - right_shoulder.z)
    if shoulder_rotation < 0.1:
        feedback.append("Increase body rotation: Rotate your body more with each stroke for better efficiency")
    else:
        positive_feedback.append("Good body rotation: You're rotating well with each stroke, which helps with efficiency")

    return feedback, positive_feedback

def analyze_butterfly(landmarks):
    feedback = []
    positive_feedback = []

    # Extract relevant landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Body undulation analysis
    shoulder_hip_distance = np.linalg.norm(np.array([left_shoulder.y - left_hip.y, right_shoulder.y - right_hip.y]))
    if shoulder_hip_distance < 0.2:
        feedback.append("Increase body undulation: Use more powerful dolphin-like movements")
    else:
        positive_feedback.append("Good body undulation: Your dolphin-like movements are effective")

    # Arm synchronization analysis
    if abs(left_shoulder.y - right_shoulder.y) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Synchronize arm movements: Keep both arms moving together")
    else:
        positive_feedback.append("Good arm synchronization: Your arms are moving together well")

    # Arm recovery analysis
    if abs(left_elbow.z - right_elbow.z) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Improve arm recovery: Keep your arms relaxed and follow a straight path over the water")
    else:
        positive_feedback.append("Good arm recovery: Your arms are following a good path over the water")

    # Kick analysis
    kick_amplitude = abs(left_ankle.y - right_ankle.y)
    if kick_amplitude < 0.3:
        feedback.append("Increase kick power: Use stronger, more rhythmic dolphin kicks")
    else:
        positive_feedback.append("Powerful kick: Your dolphin kicks are strong and rhythmic")

    # Breathing technique analysis
    if abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) > BODY_ALIGNMENT_THRESHOLD:
        feedback.append("Improve breathing technique: Keep your head movement minimal and aligned with your body")
    else:
        positive_feedback.append("Good breathing technique: Your head is well-aligned during breathing")

    # Pull pattern analysis
    if abs(left_elbow.x - right_elbow.x) > ARM_MOVEMENT_THRESHOLD:
        feedback.append("Improve pull pattern: Keep your pulls symmetrical and avoid crossing the centerline")
    else:
        positive_feedback.append("Effective pull pattern: Your arm pulls are symmetrical and efficient")

    return feedback, positive_feedback