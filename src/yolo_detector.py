import os
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch
try:
    from PIL import Image
except Exception:
    Image = None


class DetectionBox:
    """
    Lightweight container for a single detection result

    Rich comments: This object carries the bounding box, score, label
    and an optional segmentation mask produced by the model
    """

    def __init__(
        self, xyxy: Tuple[float, float, float, float], score: float, label: str, mask: Optional[np.ndarray] = None
    ):
        # Rich comment: xyxy are pixel coordinates (x1, y1, x2, y2), mask is optional HxW bool array
        self.xyxy = xyxy
        self.score = score
        self.label = label
        self.mask = mask


class YOLOEDetector:
    """
    Promptable YOLOE detector focused on spotting drones

    Rich comments: The model is loaded lazily on first use. We configure the
    text prompt to "drone" per the Ultralytics YOLOE docs so detections focus
    on that class. The API accepts numpy RGB images and returns DetectionBox objects.
    """

    def __init__(
        self,
        model_name: str = "yoloe-11s-seg.pt",
        device: Optional[str] = None,
        conf: float = 0.001,
        iou: float = 0.3,
        imgsz: int = 1280,
        visual_ref_path: Optional[str] = "assets/images/drone.png",
    ):
        # Rich comment: Store configuration and set up concurrency primitives
        self._model_name = model_name
        self._device = device
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._visual_ref_path = visual_ref_path
        self._model = None
        self._lock = threading.Lock()
        # Rich comment: Use the best performing prompt discovered by probe script
        self._prompt_labels = ["drone"]

    def _ensure_model(self) -> None:
        # Rich comment: Thread-safe lazy model initialization with text prompt configuration
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from ultralytics import YOLO
            from ultralytics.utils.downloads import attempt_download_asset

            model = YOLO(self._model_name)

            # Rich comment: We generate text prompt embeddings on CPU to avoid MPS float64 issues,
            # then move the detection model to the preferred device for inference
            preferred_device = self._device
            try:
                model.to("cpu")
            except Exception as e:
                raise RuntimeError(f"Failed moving YOLOE model to CPU for prompt embedding: {e}")

            # Rich comment: Ensure MobileCLIP asset for text prompts is fully available and not corrupt
            asset_name = "mobileclip_blt.ts"
            asset_path = attempt_download_asset(asset_name)
            if not os.path.exists(asset_path) or os.path.getsize(asset_path) < 500 * 1024 * 1024:
                # Delete partial file and retry once
                if os.path.exists(asset_path):
                    os.remove(asset_path)
                asset_path = attempt_download_asset(asset_name)
            if not os.path.exists(asset_path) or os.path.getsize(asset_path) < 500 * 1024 * 1024:
                size = os.path.getsize(asset_path) if os.path.exists(asset_path) else 0
                raise RuntimeError(
                    f"MobileCLIP asset '{asset_name}' invalid (size={size} bytes). Delete and retry or check network."
                )

            # Rich comment: Compute prompt embeddings explicitly on CPU (float32).
            # Prefer visual prompt; fallback to text.
            torch.set_default_dtype(torch.float32)
            names = self._prompt_labels
            embeddings = None

            # Attempt visual prompt using a reference drone image if available
            if self._visual_ref_path and os.path.exists(self._visual_ref_path) and Image is not None:
                try:
                    ref_img = Image.open(self._visual_ref_path).convert("RGB")
                    ref_np = np.array(ref_img)
                    if hasattr(model, "get_visual_pe"):
                        embeddings = model.get_visual_pe([ref_np])
                        names = ["drone"]
                    elif hasattr(model, "get_image_pe"):
                        embeddings = model.get_image_pe([ref_np])
                        names = ["drone"]
                except Exception:
                    embeddings = None

            if embeddings is None:
                embeddings = model.get_text_pe(names)

            model.set_classes(names, embeddings)

            # Rich comment: After successful prompt setup, move model to preferred device for inference
            if preferred_device:
                model.to(preferred_device)

            # Rich comment: Emit one-time debug about model device and prompts to verify configuration
            try:
                model_device = next(model.model.parameters()).device  # type: ignore[attr-defined]
            except Exception:
                model_device = "unknown"
            msg = (
                f"YOLOE ready | device={model_device} | prompts={names} | "
                f"imgsz={self._imgsz} conf={self._conf} iou={self._iou}"
            )
            print(msg)

            self._model = model

    def predict(self, image_rgb: np.ndarray) -> List[DetectionBox]:
        """
        Run detection on a single RGB numpy image and return filtered drone detections

        Rich comments: The input must be an HxWx3 uint8 RGB image. The method returns
        bounding boxes and optional masks, filtering to the configured prompt labels.
        """
        if image_rgb is None or image_rgb.size == 0:
            return []

        self._ensure_model()

        # Rich comment: Use the RGB numpy image directly, matching the probe script behavior
        image_input = np.ascontiguousarray(image_rgb)

        # Rich comment: Debug input frame characteristics (rate-limited)
        if not hasattr(self, "_dbg_frame_count"):
            self._dbg_frame_count = 0
        self._dbg_frame_count += 1
        if self._dbg_frame_count % 15 == 1:
            dbg = (
                f"Predict call | input_shape={image_input.shape} dtype={image_input.dtype} "
                f"conf={self._conf} iou={self._iou} imgsz={self._imgsz}"
            )
            print(dbg)

        # Rich comment: Removed saving of per-frame debug PNG to avoid disk churn during video export

        # Rich comment: Run multi-scale and crop-augmented inference to improve recall on small targets
        H, W, _ = image_input.shape
        candidates = []
        # original
        candidates.append((image_input, 1.0, 0, 0, W, H))
        # scale-up 1.5x if PIL present
        if Image is not None:
            try:
                up = np.array(Image.fromarray(image_input).resize((int(W * 1.5), int(H * 1.5))))
                candidates.append((up, 1.5, 0, 0, W, H))
            except Exception:
                pass
            # centered crop at 60% area
            cw, ch = int(W * 0.6), int(H * 0.6)
            ox = (W - cw) // 2
            oy = (H - ch) // 2
            try:
                crop = image_input[oy:oy + ch, ox:ox + cw]
                candidates.append((crop, 1.0, ox, oy, cw, ch))
            except Exception:
                pass

        aggregated: List[DetectionBox] = []

        for img, scale, ox, oy, ww, hh in candidates:
            res = self._model.predict(img, conf=self._conf, iou=self._iou, imgsz=self._imgsz, verbose=False)
            if not res:
                continue
            r = res[0]
            if not hasattr(r, "boxes") or r.boxes is None or not hasattr(r.boxes, "xyxy"):
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.zeros((0,), dtype=float)
            clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.zeros((0,), dtype=int)
            names = r.names if hasattr(r, "names") else {}

            # Rich comment: Map boxes from candidate image space back to original frame space
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                if scale != 1.0:
                    x1 /= scale
                    y1 /= scale
                    x2 /= scale
                    y2 /= scale
                x1 += ox
                x2 += ox
                y1 += oy
                y2 += oy
                label = names.get(int(clss[i]), str(int(clss[i]))) if isinstance(names, dict) else str(int(clss[i]))
                score = float(confs[i])
                aggregated.append(DetectionBox((x1, y1, x2, y2), score, label, None))

            # Rich comment: If we already have detections from this candidate, we can stop early
            if aggregated:
                break

        # Rich comment: Emit a compact debug summary
        try:
            model_device = next(self._model.model.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            model_device = "unknown"
        if not aggregated or self._dbg_frame_count % 15 == 1:
            top_conf = [] if not aggregated else sorted([b.score for b in aggregated], reverse=True)[:5]
            print(f"Det summary | n={len(aggregated)} top_conf={top_conf} device={model_device}")

        return aggregated
