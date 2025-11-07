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
    (2, 1), (3, 1),  # eyes to nose
    (4, 2), (5, 3),  # ears to eyes
    (6, 1), (7, 1),  # shoulders to nose (center connection)
    (8, 6), (9, 7),  # elbows to shoulders
    (10, 8), (11, 9),  # wrists to elbows
    (12, 10), (13, 11),  # pinky to wrists
    (14, 10), (15, 11),  # index to wrists
    (16, 6), (17, 7),  # hips to shoulders
    (18, 16), (19, 17),  # knees to hips
    (20, 18), (21, 19),  # ankles to knees
    (22, 20), (23, 21),  # heels to ankles
    (24, 20), (25, 21),  # feet to ankles
)
