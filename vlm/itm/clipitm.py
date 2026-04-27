import os
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    import clip
except Exception:
    print("Could not import clip. This is OK if you are only using the client.")


class CLIPITM:
    """CLIP-based image-text similarity model."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        self.model.eval()

    def cosine(self, image: np.ndarray, txt: str) -> float:
        image_features, text_features, _ = self._encode(image, txt)
        return float(torch.matmul(image_features, text_features.T).item())

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        cosine = self.cosine(image, txt)
        return float((cosine + 1.0) / 2.0)

    def infer(self, image: np.ndarray, txt: str) -> tuple[float, float, dict[str, float]]:
        image_features, text_features, timing = self._encode(image, txt)
        cosine = float(torch.matmul(image_features, text_features.T).item())
        return cosine, float((cosine + 1.0) / 2.0), timing

    def _encode(
        self, image: np.ndarray, txt: str
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        preprocess_start = time.perf_counter()
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([txt]).to(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        with torch.inference_mode():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        model_inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        return image_features, text_features, {
            "preprocess_ms": preprocess_ms,
            "model_inference_ms": model_inference_ms,
            "postprocess_ms": postprocess_ms,
        }


class CLIPITMClient:
    def __init__(self, port: int = 12182, route_name: str = "clipitm"):
        self.url = f"http://localhost:{port}/{route_name}"

    def infer(self, image: np.ndarray, txt: str) -> dict:
        request_start = time.perf_counter()
        response = send_request(self.url, image=image, txt=txt)
        timing = dict(response.get("timing", {}))
        timing["client_total_ms"] = (time.perf_counter() - request_start) * 1000.0
        response["timing"] = timing
        return response

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = self.infer(image, txt)
        return float(response["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        response = self.infer(image, txt)
        return float(response["itm score"])


def build_server(route_name: str = "clipitm"):
    class CLIPITMServer(ServerMixin, CLIPITM):
        def process_payload(self, payload: dict) -> dict:
            request_start = time.perf_counter()
            image = str_to_image(payload["image"])
            cosine, itm_score, timing = self.infer(image, payload["txt"])
            timing["server_total_ms"] = (time.perf_counter() - request_start) * 1000.0
            return {
                "response": cosine,
                "itm score": itm_score,
                "timing": timing,
            }

    return CLIPITMServer, route_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--model-name", type=str, default=os.environ.get("CLIP_MODEL_NAME", "ViT-B/32"))
    args = parser.parse_args()

    print("Loading CLIP model...")
    CLIPITMServer, route_name = build_server(route_name="clipitm")
    model = CLIPITMServer(model_name=args.model_name)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(model, name=route_name, port=args.port)
