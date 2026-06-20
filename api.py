from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
from pathlib import Path
import dataclasses

from utils import parse_json_to_pose12, download_video, extract_frame_as_image
from compare_service import PoseComparator
from report_service import ReportGenerator

app = FastAPI(title="Compare Pose API", description="API chấm điểm tư thế yoga/thể dục")
output_dir_path = Path("d:/compare-pose-project/output")
output_dir_path.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir_path)), name="output")

class CompareRequest(BaseModel):
    ref_url: str
    ref_height: int
    stu_url: str
    stu_height: int
    submission_data: Dict[str, Any]
    n_keyframes: Optional[int] = 30
    tolerance: Optional[float] = 0.985
    max_errors_per_pair: Optional[int] = 3

@app.post("/compare-pose")
def compare_pose(req: CompareRequest):
    try:
        output_dir = Path("d:/compare-pose-project/output")
        cache_root = Path("d:/compare-pose-project/cache")
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

        print("[API] Đang xử lý dữ liệu JSON...")
        ref_pose12_seq = parse_json_to_pose12(req.submission_data, data_key="standardData")
        stu_pose12_seq = parse_json_to_pose12(req.submission_data, data_key="studentData")

        if len(ref_pose12_seq) == 0 or len(stu_pose12_seq) == 0:
            raise HTTPException(status_code=400, detail="Dữ liệu JSON không chứa frames hợp lệ.")

        print(f"[API] Đã đọc xong JSON. Tải video... (nếu cần)")
        ref_video_path = download_video(req.ref_url, cache_root)
        stu_video_path = download_video(req.stu_url, cache_root)

        print(f"[API] Bắt đầu so sánh...")
        report_data = PoseComparator.compare_videos(
            ref_video=ref_video_path,
            ref_height=req.ref_height,
            ref_pose12_seq=ref_pose12_seq,
            stu_video=stu_video_path,
            stu_height=req.stu_height,
            stu_pose12_seq=stu_pose12_seq,
            n_keyframes=req.n_keyframes,
            tolerance=req.tolerance,
        )

        # Xuất ảnh và Report HTML
        timestamp = int(time.time())
        output_html = output_dir / f"report_{timestamp}.html"
        images_dir = output_dir / f"report_{timestamp}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_refs = set()
        extracted_stus = set()
        
        for error in report_data.errors:
            ref_frame = error["ref_frame"]
            stu_frame = error["student_frame"]
            
            ref_img_name = f"ref_frame_{ref_frame}.jpg"
            stu_img_name = f"stu_frame_{stu_frame}.jpg"
            
            if ref_frame not in extracted_refs:
                extract_frame_as_image(ref_video_path, ref_frame, images_dir / ref_img_name)
                extracted_refs.add(ref_frame)
                
            if stu_frame not in extracted_stus:
                extract_frame_as_image(stu_video_path, stu_frame, images_dir / stu_img_name)
                extracted_stus.add(stu_frame)
                
            error["ref_image_path"] = f"/output/report_{timestamp}_images/{ref_img_name}"
            error["stu_image_path"] = f"/output/report_{timestamp}_images/{stu_img_name}"
        
        ReportGenerator.render_html_report(
            ref_url=req.ref_url,
            stu_url=req.stu_url,
            report_data=report_data,
            output_path=output_html,
            max_errors_per_pair=req.max_errors_per_pair,
        )

        with open(output_html, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        return HTMLResponse(content=html_content)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
