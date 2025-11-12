ntu_pairs = (
    (2, 3), (2, 5), (3, 4), (5, 6), (6, 7), (2, 8), (8, 9),(8,11),
                    (9, 10), (5, 11), (11, 12), (12, 13), (1, 14), (1, 15), (1, 1), 
                    (14, 16), (15, 17)
)

# MediaPipe pose landmarks (25 keypoints - 1-indexed)
# 1:nose, 2-3:eyes(L,R), 4-5:ears(L,R), 6-7:shoulders(L,R), 8-9:elbows(L,R), 
# 10-11:wrists(L,R), 12-13:pinky(L,R), 14-15:index(L,R), 16-17:hips(L,R),
# 18-19:knees(L,R), 20-21:ankles(L,R), 22-23:heels(L,R), 24-25:feet(L,R)
mediapipe_pairs = (
    # Face connections
    (1, 2), (1, 3),  # nose to eyes
    (2, 4), (3, 5),  # eyes to ears
    # Upper body connections
    (6, 7),  # shoulders connection
    (1, 6), (1, 7),  # nose to shoulders
    (6, 8), (8, 10),  # left arm: shoulder -> elbow -> wrist
    (7, 9), (9, 11),  # right arm: shoulder -> elbow -> wrist
    (10, 12), (10, 14),  # left wrist to pinky and index
    (11, 13), (11, 15),  # right wrist to pinky and index
    # Torso connections
    (6, 16), (7, 17),  # shoulders to hips
    (16, 17),  # hips connection
    # Leg connections
    (16, 18), (18, 20),  # left leg: hip -> knee -> ankle
    (17, 19), (19, 21),  # right leg: hip -> knee -> ankle
    # Feet connections
    (20, 22), (20, 24),  # left ankle to heel and foot
    (21, 23), (21, 25),  # right ankle to heel and foot
)
