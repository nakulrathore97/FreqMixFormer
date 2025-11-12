ntu_pairs = (
    (2, 3), (2, 5), (3, 4), (5, 6), (6, 7), (2, 8), (8, 9),(8,11),
                    (9, 10), (5, 11), (11, 12), (12, 13), (1, 14), (1, 15), (1, 1), 
                    (14, 16), (15, 17)
)

# MediaPipe Pose bone pairs (25 joints, 1-indexed)
# 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists,
# 11-12: pinkies, 13-14: indexes, 15-16: hips, 17-18: knees, 19-20: ankles,
# 21-22: heels, 23-24: feet
mediapipe_pairs = (
    (1, 2), (2, 3), (1, 3), (3, 4), (2, 5),  # Face connections
    (5, 6), (6, 7),  # Shoulders
    (1, 6), (1, 7),  # Nose to shoulders
    (6, 8), (8, 10),  # Left arm: shoulder -> elbow -> wrist
    (7, 9), (9, 11),  # Right arm: shoulder -> elbow -> wrist
    (10, 12), (10, 14),  # Left wrist to pinky and index
    (11, 13), (11, 15),  # Right wrist to pinky and index
    (6, 16), (7, 17),  # Shoulders to hips
    (16, 17),  # Hips
    (16, 18), (18, 20),  # Left leg: hip -> knee -> ankle
    (17, 19), (19, 21),  # Right leg: hip -> knee -> ankle
    (20, 22), (20, 24),  # Left ankle to heel and foot
    (21, 23), (21, 25),  # Right ankle to heel and foot
)
