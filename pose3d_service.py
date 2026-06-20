import numpy as np
from pathlib import Path
from config import Config

class Pose3DService:
    @staticmethod
    def run_tcpformer(pose2d: np.ndarray, ckpt_path: Path, device: str) -> np.ndarray:
        if len(pose2d) == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)

        import torch
        from tcpformer_model import MemoryInducedTransformer

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint TCPFormer: {ckpt_path}")

        model = MemoryInducedTransformer(
            n_layers=14,
            dim_in=3,
            dim_feat=128,
            dim_rep=512,
            dim_out=3,
            n_frames=81
        )

        print(f"[tcpformer] Loading checkpoint: {ckpt_path.name}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        
        device_obj = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        model = model.to(device_obj)
        model.eval()

        F = len(pose2d)
        pose3d_out = np.zeros((F, 17, 3), dtype=np.float32)

        chunk_size = 81
        with torch.no_grad():
            for start_idx in range(0, F, chunk_size):
                end_idx = min(start_idx + chunk_size, F)
                chunk = pose2d[start_idx:end_idx].copy()

                pelvis = chunk[:, 0:1, :2].copy()
                chunk[:, :, :2] -= pelvis

                if len(chunk) < chunk_size:
                    pad_len = chunk_size - len(chunk)
                    pad_frames = np.repeat(chunk[-1:], pad_len, axis=0)
                    chunk_padded = np.concatenate([chunk, pad_frames], axis=0)
                else:
                    chunk_padded = chunk

                x_tensor = torch.from_numpy(chunk_padded).unsqueeze(0).to(device_obj)
                y_tensor = model(x_tensor)
                
                y_np = y_tensor.squeeze(0).cpu().numpy()
                
                pose3d_out[start_idx:end_idx] = y_np[: (end_idx - start_idx)]

        return pose3d_out

    @staticmethod
    def pose12_from_tcpformer(pose3d: np.ndarray, frame_id: int) -> np.ndarray:
        if frame_id < 0 or frame_id >= len(pose3d):
            raise IndexError(f"Frame {frame_id} nằm ngoài pose3d shape={pose3d.shape}")
        
        return pose3d[frame_id][Config.H36M_TO_12, :3].astype(np.float32)
