# dto.py
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ErrorDetail:
    bone: str
    child_joint: str
    cosine: float
    move_cm: float
    pair_index: int = 0
    ref_frame: int = 0
    student_frame: int = 0
    ref_image_path: str = ""
    stu_image_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bone": self.bone,
            "child_joint": self.child_joint,
            "cosine": self.cosine,
            "move_cm": self.move_cm,
            "pair_index": self.pair_index,
            "ref_frame": self.ref_frame,
            "student_frame": self.student_frame,
            "ref_image_path": self.ref_image_path,
            "stu_image_path": self.stu_image_path,
        }

@dataclass
class CompareReport:
    ref_height: int
    stu_height: int
    ref_keyframes: List[int]
    stu_keyframes: List[int]
    m: int
    errors: List[Dict[str, Any]]
    use_pkl_joints: bool = False
