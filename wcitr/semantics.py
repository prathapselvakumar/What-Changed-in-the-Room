"""Step 4b: name each changed object and double-check it with a vision model.

For every object, the frame that sees most of it (depth-verified) is chosen
and a crop is cut from the full-resolution keyframe (in memory). The same 3D region
is also cropped from the other video, so a before/after pair exists for every change.

  * CLIP: image embedding (used for matching) + offline label fallback
  * Claude (optional): short label, description, and whether the change is
    really visible in the before/after pair. Text is generated only from these
    crops and the geometry, so it cannot contradict the 3D result.
"""

import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from .geometry import project

VOCAB = [
    "chair", "office chair", "stool", "table", "desk", "sofa", "bed", "pillow", "cushion", "blanket",
    "backpack", "bag", "suitcase", "box", "cardboard box", "basket", "bin", "trash can", "laundry basket",
    "lamp", "plant", "potted plant", "vase", "bottle", "water bottle", "cup", "mug", "bowl", "plate",
    "drinking glass", "glass", "jug",
    "book", "notebook", "laptop", "monitor", "keyboard", "mouse", "phone", "tablet", "speaker", "headphones",
    "television", "remote control", "shoe", "shoes", "jacket", "shirt", "towel", "umbrella", "helmet",
    "toy", "stuffed animal", "ball", "guitar", "fan", "heater", "clock", "picture frame", "mirror",
    "shelf", "cabinet", "drawer", "door", "curtain", "rug", "mat", "broom", "bucket", "ladder",
    "toolbox", "printer", "router", "cable", "charger", "power strip", "bicycle", "kettle", "food container",
]


class Labeler:
    def __init__(self, cfg, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.cfg, self.device = cfg, device
        self.clip = CLIPModel.from_pretrained(cfg["clip_model"]).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(cfg["clip_model"])
        with torch.no_grad():
            txt = self.proc(text=[f"a photo of a {w}" for w in VOCAB], return_tensors="pt", padding=True).to(device)
            t = self._features(self.clip.get_text_features(**txt))
        self.text_emb = t / t.norm(dim=-1, keepdim=True)
        self.client = None
        if cfg.get("use_claude", True):
            import anthropic
            try:
                self.client = anthropic.Anthropic()
                if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                    self.client.models.retrieve(cfg["model"])  # checks a stored `ant auth login` profile
            except (anthropic.AnthropicError, TypeError) as e:
                print("[semantics] No working Claude credentials (set ANTHROPIC_API_KEY to enable); "
                      f"using CLIP labels only. ({type(e).__name__})")
                self.client = None

    @staticmethod
    def _features(out):
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def embed(self, img):
        x = self.proc(images=img, return_tensors="pt").to(self.device)
        e = self._features(self.clip.get_image_features(**x))[0]
        return e / e.norm()

    def clip_label(self, emb):
        probs = (100 * (self.text_emb @ emb)).softmax(0)
        k = int(probs.argmax())
        return VOCAB[k], float(probs[k])

    def ask_claude(self, crop_change, crop_other, kind):
        """kind: 'removed' (object in before, gone after) or 'added' (absent before, present after)."""
        if self.client is None or crop_other is None:
            return None
        import anthropic

        first, second = ("BEFORE", "AFTER") if kind == "removed" else ("AFTER", "BEFORE")
        prompt = (
            f"A 3D change detector compared two videos of a room. Image 1 is from the {first} video and shows "
            f"a region where it found an object that is {'gone' if kind == 'removed' else 'new'}. "
            f"Image 2 shows the same 3D region in the {second} video (the viewpoint may differ).\n"
            "1. Name the main object in the centre of image 1 with a short common noun (e.g. 'chair', 'backpack').\n"
            "2. Describe it in at most 12 words (colour, material, notable parts).\n"
            f"3. Is that object visible in image 1 but missing from that spot in image 2? Answer false if it is "
            "clearly still there, or if image 1 shows no distinct object (only wall, floor, reflection, shadow)."
        )
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "description": {"type": "string"},
                "change_visible": {"type": "boolean"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["label", "description", "change_visible", "confidence"],
            "additionalProperties": False,
        }
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(c)}}
                   for c in (crop_change, crop_other)]
        content.append({"type": "text", "text": prompt})
        try:
            resp = self.client.beta.messages.create(
                model=self.cfg["model"],
                max_tokens=2000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIStatusError as e:
            print(f"[semantics] Claude request failed ({e.status_code}); using CLIP label.")
            return None
        except anthropic.APIConnectionError:
            print("[semantics] Claude unreachable; using CLIP label.")
            return None
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            return json.loads(text) if text else None
        except json.JSONDecodeError:
            return None


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


@torch.no_grad()
def best_view(xyz, frames, recon, require_visible=True, device="cuda"):
    """Frame among `frames` that sees most of the points, plus their pixel box (model resolution)."""
    E = torch.as_tensor(recon["extrinsic"][frames], device=device)
    K = torch.as_tensor(recon["intrinsic"][frames], device=device)
    D = torch.as_tensor(recon["depth"][frames], device=device)
    X = torch.as_tensor(xyz, device=device, dtype=torch.float32)
    T, H, W = D.shape
    u, v, z = project(X, E, K)
    ui, vi = u.round().long(), v.round().long()
    inb = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if require_visible:
        d = D[torch.arange(T, device=device)[:, None], vi.clamp(0, H - 1), ui.clamp(0, W - 1)]
        inb &= (d - z).abs() <= torch.clamp(0.05 * z, min=0.05)
    # Prefer views where the object is near the image centre.
    cx = (u.clamp(0, W) - W / 2).abs() / W + (v.clamp(0, H) - H / 2).abs() / H
    score = inb.sum(1).float() * (1.0 - 0.5 * (cx * inb).sum(1) / inb.sum(1).clamp_min(1))
    k = int(score.argmax())
    if inb[k].sum() < 10:
        return None
    uu, vv = u[k][inb[k]].cpu().numpy(), v[k][inb[k]].cpu().numpy()
    box = (np.percentile(uu, 1), np.percentile(vv, 1), np.percentile(uu, 99), np.percentile(vv, 99))
    return {"frame": int(frames[k]), "box": box, "n_visible": int(inb[k].sum())}


def crop(keyframes, recon, view, margin=0.25, min_px=96):
    """Cut the view's box out of the full-resolution keyframe (in memory)."""
    img = Image.fromarray(keyframes[view["frame"]])
    s, ox, oy = recon["mapping"][view["frame"]]
    x0, y0, x1, y1 = [(c - o) / s for c, o in zip(view["box"], (ox, oy, ox, oy))]
    w, h = max(x1 - x0, min_px), max(y1 - y0, min_px)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = w * (1 + 2 * margin), h * (1 + 2 * margin)
    box = (max(0, cx - w / 2), max(0, cy - h / 2), min(img.width, cx + w / 2), min(img.height, cy + h / 2))
    return img.crop(tuple(int(round(b)) for b in box))


def label_objects(objects, kind, recon, keyframes, labeler):
    """Adds embedding, label, description and verification to each object (in place).

    keyframes: full-resolution RGB keyframes in the same order as the reconstruction.
    Crops stay in memory; nothing is written to disk.
    """
    vid = 0 if kind == "removed" else 1
    own = np.nonzero(recon["video"] == vid)[0]
    other = np.nonzero(recon["video"] != vid)[0]
    for obj in objects:
        view = best_view(obj["xyz"], own, recon) or best_view(obj["xyz"], own, recon, require_visible=False)
        ctx_view = best_view(obj["xyz"], other, recon, require_visible=False)
        c_obj = crop(keyframes, recon, view) if view else None
        c_ctx = crop(keyframes, recon, ctx_view) if ctx_view else None
        obj["view_frame"] = view["frame"] if view else None
        if c_obj is not None:
            obj["embedding"] = labeler.embed(c_obj).cpu().numpy()
            if not obj.get("label_fixed"):
                obj["label"], obj["label_score"] = labeler.clip_label(
                    torch.as_tensor(obj["embedding"], device=labeler.device))
                obj["label_source"] = "clip"
        answer = labeler.ask_claude(c_obj, c_ctx, kind) if c_obj is not None else None
        if answer:
            obj["label"] = answer["label"].strip().lower()
            obj["description"] = answer["description"].strip()
            obj["verified"] = bool(answer["change_visible"])
            obj["verify_confidence"] = answer["confidence"]
            obj["label_source"] = "claude"
    return objects
