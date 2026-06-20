from typing import Optional, List, Iterable, Tuple
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from config import Config

class Pose2DService:
    @staticmethod
    def weighted_point(kps: np.ndarray, names: Iterable[str | int]) -> np.ndarray:
        indices = [Config.COCO[x] if isinstance(x, str) else int(x) for x in names]
        pts = kps[indices].astype(np.float32)
        conf = pts[:, 2]
        valid = np.isfinite(pts).all(axis=1) & (conf > 0)
        if not np.any(valid):
            return np.zeros(3, dtype=np.float32)
        pts, conf = pts[valid], conf[valid]
        w = conf / max(float(conf.sum()), 1e-6)
        xy = (pts[:, :2] * w[:, None]).sum(axis=0)
        return np.array([xy[0], xy[1], float(conf.mean())], dtype=np.float32)

    @staticmethod
    def weighted_existing(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        conf = pts[:, 2]
        valid = np.isfinite(pts).all(axis=1) & (conf > 0)
        if not np.any(valid):
            return np.zeros(3, dtype=np.float32)
        pts, conf = pts[valid], conf[valid]
        w = conf / max(float(conf.sum()), 1e-6)
        xy = (pts[:, :2] * w[:, None]).sum(axis=0)
        return np.array([xy[0], xy[1], float(conf.mean())], dtype=np.float32)

    @staticmethod
    def coco_to_h36m(coco_kps: Optional[np.ndarray]) -> np.ndarray:
        """Convert COCO 17 keypoints của YOLO sang H36M 17 joints dạng (17,3)."""
        h36m = np.zeros((17, 3), np.float32)
        if coco_kps is None:
            return h36m

        if coco_kps.shape[-1] == 2:
            coco_kps = np.concatenate(
                [coco_kps.astype(np.float32), np.ones((coco_kps.shape[0], 1), np.float32)], axis=1
            )
        coco_kps = coco_kps.astype(np.float32)

        pelvis = Pose2DService.weighted_point(coco_kps, ["left_hip", "right_hip"])
        shoulder_c = Pose2DService.weighted_point(coco_kps, ["left_shoulder", "right_shoulder"])
        head = Pose2DService.weighted_point(coco_kps, ["nose", "left_eye", "right_eye", "left_ear", "right_ear"])

        h36m[0] = pelvis
        h36m[1] = Pose2DService.weighted_point(coco_kps, ["right_hip"])
        h36m[2] = Pose2DService.weighted_point(coco_kps, ["right_knee"])
        h36m[3] = Pose2DService.weighted_point(coco_kps, ["right_ankle"])
        h36m[4] = Pose2DService.weighted_point(coco_kps, ["left_hip"])
        h36m[5] = Pose2DService.weighted_point(coco_kps, ["left_knee"])
        h36m[6] = Pose2DService.weighted_point(coco_kps, ["left_ankle"])
        h36m[7] = Pose2DService.weighted_existing(np.stack([pelvis, shoulder_c]))
        h36m[8] = shoulder_c
        h36m[9] = Pose2DService.weighted_existing(np.stack([shoulder_c, head]))
        h36m[10] = head
        h36m[11] = Pose2DService.weighted_point(coco_kps, ["left_shoulder"])
        h36m[12] = Pose2DService.weighted_point(coco_kps, ["left_elbow"])
        h36m[13] = Pose2DService.weighted_point(coco_kps, ["left_wrist"])
        h36m[14] = Pose2DService.weighted_point(coco_kps, ["right_shoulder"])
        h36m[15] = Pose2DService.weighted_point(coco_kps, ["right_elbow"])
        h36m[16] = Pose2DService.weighted_point(coco_kps, ["right_wrist"])
        return h36m

    @staticmethod
    def pick_best_person(result: Any) -> Optional[np.ndarray]:
        if result.keypoints is None or result.keypoints.data is None:
            return None
        data = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
        if data.size == 0:
            return None
        if data.shape[-1] == 2:
            conf = np.ones(data.shape[:2] + (1,), np.float32)
            data = np.concatenate([data, conf], axis=-1)
        scores = np.nanmean(data[:, :, 2], axis=1)
        if not np.isfinite(scores).any():
            return None
        return data[int(np.nanargmax(scores))]

    @staticmethod
    def process_video_to_h36m(video_path: Path, model_name: str, img_size: int, conf_thres: float, device: str) -> np.ndarray:
        from tqdm.auto import tqdm
        from ultralytics import YOLO
        from utils import count_video_frames

        print(f"[pose2d] Running YOLO: {video_path.name}")
        model = YOLO(model_name)
        total = count_video_frames(video_path)
        frames: List[np.ndarray] = []

        stream = model.predict(
            source=str(video_path),
            stream=True,
            imgsz=img_size,
            conf=conf_thres,
            device=device,
            verbose=False,
        )

        for result in tqdm(stream, total=total, desc=video_path.name, leave=False):
            person = Pose2DService.pick_best_person(result)
            frames.append(Pose2DService.coco_to_h36m(person))

        if not frames:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return np.stack(frames).astype(np.float32)

    @staticmethod
    def extract_key_frames(pose_array: np.ndarray, n_clusters: int = 30) -> np.ndarray:
        F = pose_array.shape[0]
        if F == 0:
            return np.array([], dtype=np.int64)

        mean_conf = pose_array[:, :, 2].mean(axis=1)
        valid_indices = np.where(mean_conf > 0)[0]

        if len(valid_indices) == 0:
            step = max(1, F // n_clusters)
            return np.arange(0, F, step)[:n_clusters].astype(np.int64)

        xy = pose_array[valid_indices, :, :2].astype(np.float32)
        pelvis = xy[:, 0:1, :]
        xy_norm = xy - pelvis
        vectors = xy_norm.reshape(len(valid_indices), -1)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        k = min(n_clusters, len(valid_indices))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(vectors)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        rep_valid_indices: List[int] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not mask.any():
                continue
            cluster_vectors = vectors[mask]
            cluster_orig_idx = np.where(mask)[0]
            centroid = centroids[cluster_id]
            dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
            best = cluster_orig_idx[int(np.argmin(dists))]
            rep_valid_indices.append(int(valid_indices[best]))

        return np.array(sorted(set(rep_valid_indices)), dtype=np.int64)

    @staticmethod
    def get_or_create_pose_and_keyframes(
        video_path: Path, cache_dir: Path, model_name: str, img_size: int, 
        conf_thres: float, device: str, n_keyframes: int, overwrite: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pose_path = cache_dir / f"{video_path.stem}_h36m_yolo.npy"
        key_path = cache_dir / f"{video_path.stem}_keyframes.npy"

        if pose_path.exists() and not overwrite:
            pose = np.load(pose_path)
            print(f"[cache] pose2d: {pose_path}")
        else:
            pose = Pose2DService.process_video_to_h36m(video_path, model_name, img_size, conf_thres, device)
            np.save(pose_path, pose)
            print(f"[save] pose2d: {pose_path} shape={pose.shape}")

        if key_path.exists() and not overwrite:
            keyframes = np.load(key_path).astype(np.int64)
            print(f"[cache] keyframes: {key_path}")
        else:
            keyframes = Pose2DService.extract_key_frames(pose, n_clusters=n_keyframes)
            np.save(key_path, keyframes)
            print(f"[save] keyframes: {key_path} n={len(keyframes)}")

        return pose, keyframes
