# config.py

class Config:
    COCO = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    }

    H36M_NAMES = [
        "Root", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot", "Spine",
        "Thorax", "NeckBase", "Head", "LShoulder", "LElbow", "LWrist",
        "RShoulder", "RElbow", "RWrist",
    ]

    # 12 points according to the problem requirement
    H36M_TO_12 = [
        14, 15, 16,
        11, 12, 13,
        1, 2, 3,
        4, 5, 6,
    ]

    SMPL24_TO_12 = [
        17, 19, 21,
        16, 18, 20,
        2, 5, 8,
        1, 4, 7,
    ]

    BONES_12 = [
        (0, 1, "Tay trên phải", "Cùi chỏ phải"),
        (1, 2, "Tay dưới phải", "Cổ tay phải"),
        (3, 4, "Tay trên trái", "Cùi chỏ trái"),
        (4, 5, "Tay dưới trái", "Cổ tay trái"),
        (6, 7, "Chân trên phải", "Đầu gối phải"),
        (7, 8, "Chân dưới phải", "Mắt cá/Bàn chân phải"),
        (9, 10, "Chân trên trái", "Đầu gối trái"),
        (10, 11, "Chân dưới trái", "Mắt cá/Bàn chân trái"),
    ]

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
