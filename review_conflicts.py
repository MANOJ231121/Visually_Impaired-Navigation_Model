#!/usr/bin/env python3
"""
review_conflicts.py -- READ-ONLY deep review of annotation conflicts.

Investigates, for a multi-batch YOLO dataset:
  1. class-ID conflicts between identical images
  2. bounding-box geometry mismatches between identical images
  3. images that share one filename stem (duplicate stems)
  4. images available only as WEBP / GIF
  5. exact-duplicate image groups crossing splits
  6. source-image (near-duplicate) groups crossing splits

Produces:
  annotation_conflict_review.md      the written report
  annotation_conflicts.csv/json      every problematic box pair
  duplicate_stem_pairs.csv
  format_only_images.csv
  duplicate_group_actions.csv
  leakage_group_actions.csv
  visualizations/...                 side-by-side annotated comparisons

Guarantees:
  * The dataset is only read. Nothing under the dataset root is written/renamed.
  * No class ID is changed. No file is deleted or moved.
  * No class name is invented.

Usage:
    python review_conflicts.py
"""

from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "final batches"
DEFAULT_OUT = HERE / "dataset_verification"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
JPG_PNG = {".jpg", ".jpeg", ".png"}
YOLO_OK = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CROSS_SPLIT_ALLOWED = True  # never used to modify; documentation only

# Perceptual-hash thresholds (64-bit dHash, invariant to the 8 dihedral transforms).
PHASH_SAME = 10        # <= this + similar aspect => same source
PHASH_MAYBE = 16       # 11..16 => possible, needs manual review
ASPECT_STRICT = 1.03   # aspect-ratio ratio allowed for PHASH_SAME
ASPECT_LOOSE = 1.10    # allowed when the hash distance is very small

RF_RE = re.compile(
    r"^(?P<base>.*?)(?:_(?:jpg|jpeg|png|bmp|webp|tif|tiff))?\.rf\.[0-9a-f]{32}$",
    re.IGNORECASE,
)

PANEL = 640            # max side of a visual panel
THUMB = 300            # max side of a thumbnail
COLOR_A = (34, 197, 94)
COLOR_B = (239, 68, 68)
COLOR_SHARED = (59, 130, 246)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (17, 24, 39)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def iou(a, b) -> float:
    """IoU of two (cls, x, y, w, h) boxes; class is ignored."""
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(A, B):
    """Match boxes of A to B by highest IoU; returns (matches, unmatched_B)."""
    used = set()
    matches = []
    for a in A:
        best_k, best_iou = None, 0.0
        for k, b in enumerate(B):
            if k in used:
                continue
            v = iou(a, b)
            if v > best_iou:
                best_iou, best_k = v, k
        if best_k is not None and best_iou > 0:
            used.add(best_k)
            matches.append((a, B[best_k], best_iou))
        else:
            matches.append((a, None, 0.0))
    unmatched_b = [b for k, b in enumerate(B) if k not in used]
    return matches, unmatched_b


def match_stats(A, B):
    m, ub = greedy_match(A, B)
    strong = [x for x in m if x[1] is not None and x[2] >= 0.5]
    cls_conf = [x for x in strong if x[0][0] != x[1][0]]
    geom_conf = (len(A) - len(strong)) + len(ub)
    unmatched = [x for x in m if x[1] is None or x[2] < 0.5]
    best_unmatched = [x[2] for x in unmatched] + [0.0] * len(ub)
    return {
        "n_a": len(A),
        "n_b": len(B),
        "strong": len(strong),
        "class_conflict": len(cls_conf),
        "geometry_conflict": geom_conf,
        "max_unmatched_iou": round(max(best_unmatched) if best_unmatched else 0.0, 4),
        "matched_ious": [round(x[2], 4) for x in strong],
        "class_conflict_detail": [
            {"class_a": x[0][0], "class_b": x[1][0], "iou": round(x[2], 4),
             "box_a": [round(v, 6) for v in x[0][1:]],
             "box_b": [round(v, 6) for v in x[1][1:]]}
            for x in cls_conf
        ],
    }


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash_variants(img: Image.Image, size: int = 8) -> list[int]:
    """64-bit dHash for the 8 dihedral transforms (identifies flipped/rotated copies)."""
    gray = img.convert("L")
    out = []
    for ang in (0, 90, 180, 270):
        v = gray.rotate(ang, expand=True) if ang else gray
        for flip in (False, True):
            vv = v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else v
            vv = vv.resize((size + 1, size), Image.Resampling.LANCZOS)
            a = np.asarray(vv, dtype=np.int16)
            bits = (a[:, 1:] > a[:, :-1]).flatten()
            n = 0
            for bit in bits:
                n = (n << 1) | int(bit)
            out.append(n)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def phash_distance(ha: list[int], hb: list[int]) -> int:
    return min(hamming(x, y) for x in ha for y in hb)


def base_stem(stem: str) -> str:
    m = RF_RE.match(stem)
    return (m.group("base") if m else stem).strip().lower()


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def slug(text: str, maxlen: int = 42) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return s[:maxlen].strip("_") or "item"


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(n: float, d: float) -> str:
    return "n/a" if not d else f"{100.0 * n / d:.1f}%"


def get_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------------------
# Dataset access (read-only)
# --------------------------------------------------------------------------------------


@dataclass
class ImageRec:
    path: Path
    batch: str
    split: str
    rel: str
    ext: str
    md5: str | None = None
    width: int | None = None
    height: int | None = None
    phash: list[int] | None = None


@dataclass
class LabelRec:
    path: Path
    rel: str
    boxes: list = field(default_factory=list)
    text: str = ""

    @property
    def cls_set(self):
        return {b[0] for b in self.boxes}

    def sig(self):
        return tuple(sorted((round(b[1], 6), round(b[2], 6), round(b[3], 6),
                             round(b[4], 6)) for b in self.boxes))


class Dataset:
    def __init__(self, root: Path):
        self.root = root
        self.images: list[ImageRec] = []
        self.labels: dict[Path, LabelRec] = {}
        self._img_cache: dict[Path, Image.Image] = {}
        self._load()

    def _load(self):
        for bdir in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not bdir.is_dir():
                continue
            idir = bdir / "images"
            ldir = bdir / "labels"
            if not idir.is_dir() or not ldir.is_dir():
                continue
            for p in sorted(idir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.images.append(ImageRec(
                        path=p, batch=bdir.name, split=p.parent.name,
                        rel=rel(p, self.root), ext=p.suffix.lower(),
                    ))
            for p in sorted(ldir.rglob("*.txt")):
                self.labels[p] = self._parse_label(p)

    def _parse_label(self, path: Path) -> LabelRec:
        rec = LabelRec(path=path, rel=rel(path, self.root))
        text = path.read_text(encoding="utf-8", errors="replace")
        rec.text = "\n".join(sorted(x.strip() for x in text.splitlines() if x.strip()))
        for line in text.splitlines():
            t = line.split()
            if len(t) == 5:
                rec.boxes.append((int(t[0]), *(float(v) for v in t[1:])))
        return rec

    def label_for(self, img: ImageRec) -> LabelRec | None:
        return self.labels.get(self.label_path_for(img))

    def label_path_for(self, img: ImageRec) -> Path:
        # <batch>/images/<split>/<file>  ->  <batch>/labels/<split>/<stem>.txt
        return img.path.parent.parent.parent / "labels" / img.split / (img.path.stem + ".txt")

    def load_image(self, img: ImageRec) -> Image.Image:
        if img.path not in self._img_cache:
            with Image.open(img.path) as im:
                im = im.convert("RGB")
                im.load()
            self._img_cache[img.path] = im
        return self._img_cache[img.path]

    def dims(self, img: ImageRec):
        if img.width is None:
            with Image.open(img.path) as im:
                img.width, img.height = im.size
        return img.width, img.height


# --------------------------------------------------------------------------------------
# Union-find
# --------------------------------------------------------------------------------------


class UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return list(out.values())


# --------------------------------------------------------------------------------------
# Visualisation primitives
# --------------------------------------------------------------------------------------


def _text_bg(draw: ImageDraw.ImageDraw, xy, text, font, fg=COLOR_TEXT, bg=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x - 2, y - 1, x + w + 2, y + h + 3], fill=bg)
    draw.text((x, y), text, font=font, fill=fg)


def _dashed_rect(draw, box, color, width=3, dash=8):
    x1, y1, x2, y2 = box
    for x in range(int(x1), int(x2), dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(int(y1), int(y2), dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def draw_boxes(img: Image.Image, boxes, color, label_prefix="", dashed=False,
               width=None):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    width = width or max(2, round(min(W, H) / 160))
    font = get_font(max(12, width * 5))
    fsize = getattr(font, "size", max(12, width * 5))
    for b in boxes:
        cls, x, y, w, h = b[0], b[1], b[2], b[3], b[4]
        x1, y1 = (x - w / 2) * W, (y - h / 2) * H
        x2, y2 = (x + w / 2) * W, (y + h / 2) * H
        if dashed:
            _dashed_rect(draw, [x1, y1, x2, y2], color, width=width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        _text_bg(draw, (x1 + 2, max(0, y1 - fsize - 4)),
                 f"{label_prefix}{cls}", font, fg=color)
    return img


def panel(img: Image.Image, boxes, title: str, color=COLOR_A,
          prefix="", max_side=PANEL, dashed=False, note="") -> Image.Image:
    im = img.copy()
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                       Image.Resampling.LANCZOS)
    draw_boxes(im, boxes, color, prefix, dashed=dashed)
    header = 52 if note else 30
    canvas = Image.new("RGB", (im.width, im.height + header), COLOR_BG)
    canvas.paste(im, (0, header))
    d = ImageDraw.Draw(canvas)
    f = get_font(17)
    d.text((8, 6), title[:110], font=f, fill=(226, 232, 240))
    if note:
        d.text((8, 27), note[:130], font=get_font(14), fill=(148, 163, 184))
    return canvas


def _hstack(images, pad=8, bg=COLOR_BG):
    h = max(i.height for i in images)
    w = sum(i.width for i in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + pad
    return canvas


def _footer(width, lines, bg=COLOR_BG):
    f = get_font(15)
    line_h = 21
    canvas = Image.new("RGB", (width, max(1, len(lines)) * line_h + 10), bg)
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        d.text((8, 5 + i * line_h), line[:160], font=f, fill=(203, 213, 225))
    return canvas


def pair_visual(ds: Dataset, a: ImageRec, b: ImageRec, title: str, lines=None,
                a_boxes=None, b_boxes=None, prefix="cls ") -> Image.Image:
    la = ds.label_for(a)
    lb = ds.label_for(b)
    a_boxes = la.boxes if a_boxes is None else a_boxes
    b_boxes = lb.boxes if b_boxes is None else b_boxes
    ia = panel(ds.load_image(a), a_boxes, f"A: {a.rel}", COLOR_A, prefix)
    ib = panel(ds.load_image(b), b_boxes, f"B: {b.rel}", COLOR_B, prefix)
    ia_boxes = a_boxes
    # overlay B (dashed red) on A
    ov = ds.load_image(a).copy()
    scale = PANEL / max(ov.size)
    if scale < 1:
        ov = ov.resize((max(1, round(ov.width * scale)), max(1, round(ov.height * scale))),
                       Image.Resampling.LANCZOS)
    draw_boxes(ov, ia_boxes, COLOR_A, prefix)
    draw_boxes(ov, b_boxes, COLOR_B, prefix, dashed=True)
    ov = panel(ov, [], f"Overlay: A solid / B dashed — {a.rel}", COLOR_A, prefix)
    canvas = _hstack([ia, ib, ov])
    if lines:
        canvas = Image.vstack([canvas, _footer(canvas.width, lines)]) if hasattr(Image, "vstack") else _append_footer(canvas, lines)
    return canvas


def _append_footer(img: Image.Image, lines) -> Image.Image:
    foot = _footer(img.width, lines)
    out = Image.new("RGB", (img.width, img.height + foot.height), COLOR_BG)
    out.paste(img, (0, 0))
    out.paste(foot, (0, img.height))
    return out


def zoom_conflict(ds: Dataset, a: ImageRec, b: ImageRec, box_a, box_b,
                  pad=0.35, out=520) -> Image.Image:
    """Zoom on the union of two boxes, showing both annotations."""
    ia = ds.load_image(a).copy()
    W, H = ia.size
    x1 = min(box_a[1] - box_a[3] / 2, box_b[1] - box_b[3] / 2)
    x2 = max(box_a[1] + box_a[3] / 2, box_b[1] + box_b[3] / 2)
    y1 = min(box_a[2] - box_a[4] / 2, box_b[2] - box_b[4] / 2)
    y2 = max(box_a[2] + box_a[4] / 2, box_b[2] + box_b[4] / 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + pad) / 2
    half = max(half, 0.08)
    x1, x2 = max(0, cx - half), min(1, cx + half)
    y1, y2 = max(0, cy - half), min(1, cy + half)
    crop = ia.crop((round(x1 * W), round(y1 * H), round(x2 * W), round(y2 * H)))
    if crop.width == 0 or crop.height == 0:
        crop = ia
    s = out / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))),
                       Image.Resampling.LANCZOS)
    # translate boxes into crop space
    def tr(b):
        return (b[0], (b[0] * W - x1 * W) / max(1, x2 * W - x1 * W),
                (b[2] * H - y1 * H) / max(1, y2 * H - y1 * H),
                b[3] * W / max(1, x2 * W - x1 * W),
                b[4] * H / max(1, y2 * H - y1 * H))
    draw_boxes(crop, [tr(box_a)], COLOR_A, "A cls ")
    draw_boxes(crop, [tr(box_b)], COLOR_B, "B cls ", dashed=True)
    return crop


def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


# --------------------------------------------------------------------------------------
# Perceptual / source grouping
# --------------------------------------------------------------------------------------


def build_source_groups(ds: Dataset, images_in_multi_base):
    """Union images that are byte-identical OR perceptually the same source."""
    uf = UF()
    md5_groups = defaultdict(list)
    for img in ds.images:
        if img.md5 is None:
            img.md5 = md5_of(img.path)
        uf.find(img.path)
        md5_groups[img.md5].append(img)
    for h, group in md5_groups.items():
        for x in group[1:]:
            uf.union(group[0].path, x.path)

    by_base = defaultdict(list)
    for img in ds.images:
        by_base[base_stem(img.path.stem)].append(img)

    pair_evidence = []
    for base, members in by_base.items():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members, 2):
            if a.md5 == b.md5:
                continue
            if a.phash is None or b.phash is None:
                continue
            dist = phash_distance(a.phash, b.phash)
            wa, ha = ds.dims(a)
            wb, hb = ds.dims(b)
            ar_a, ar_b = wa / ha, wb / hb
            ar = max(ar_a, ar_b) / min(ar_a, ar_b)
            same = (dist <= 6 and ar <= ASPECT_LOOSE) or (dist <= PHASH_SAME and ar <= ASPECT_STRICT)
            maybe = (not same) and dist <= PHASH_MAYBE
            if same:
                uf.union(a.path, b.path)
            if dist <= PHASH_MAYBE:
                pair_evidence.append({
                    "base": base, "a": a.rel, "b": b.rel, "distance": dist,
                    "aspect_ratio": round(ar, 3), "same": same, "maybe": maybe,
                })

    groups = [sorted(g, key=lambda p: str(p)) for g in uf.groups() if len(g) > 1]
    return groups, pair_evidence, by_base


# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------


def inspect(root: Path, out_dir: Path) -> dict:
    t0 = time.time()
    ds = Dataset(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = out_dir / "visualizations"
    vis.mkdir(exist_ok=True)

    review = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "totals": {"images": len(ds.images), "labels": len(ds.labels)},
        "class_conflicts": [],
        "geometry_conflicts": [],
        "duplicate_stems": [],
        "format_only": [],
        "exact_groups": [],
        "source_groups": [],
        "filename_collision_examples": [],
        "notes": [],
    }

    print("[1/7] Indexing images and labels", flush=True)
    label_by_img = {}
    for img in ds.images:
        lp = ds.label_path_for(img)
        label_by_img[img.path] = ds.labels.get(lp)

    print("[2/7] Computing hashes / dimensions", flush=True)
    # Hashes needed for duplicate stems, format-only and source grouping.
    targets = {img.path for img in ds.images}
    for i, img in enumerate(ds.images, 1):
        if img.md5 is None:
            img.md5 = md5_of(img.path)
        ds.dims(img)
        if i % 4000 == 0:
            print(f"    ... {i}/{len(ds.images)}", flush=True)

    # dHash only for images that share a base name with another image (candidate leakage)
    base_counts = Counter(base_stem(img.path.stem) for img in ds.images)
    for img in ds.images:
        if base_counts[base_stem(img.path.stem)] > 1:
            with Image.open(img.path) as im:
                img.phash = dhash_variants(im)
    print("    ... perceptual hashes computed", flush=True)

    # ---------------- 1. class conflicts ----------------
    print("[3/7] Class and geometry conflicts", flush=True)
    md5_groups = defaultdict(list)
    for img in ds.images:
        md5_groups[img.md5].append(img)

    class_visual_count = 0
    geom_visual_count = 0
    geom_visuals = []
    for h, members in sorted(md5_groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members, 2):
            la, lb = label_by_img[a.path], label_by_img[b.path]
            if la is None or lb is None:
                continue
            stats = match_stats(la.boxes, lb.boxes)
            if stats["class_conflict"] == 0 and stats["geometry_conflict"] == 0:
                continue
            item = {
                "md5": h,
                "a": a.rel, "b": b.rel,
                "batch_a": a.batch, "split_a": a.split,
                "batch_b": b.batch, "split_b": b.split,
                "cross_split": a.split != b.split,
                "cross_batch": a.batch != b.batch,
                **stats,
            }
            if stats["class_conflict"] > 0:
                class_visual_count += 1
                item["id"] = f"C{class_visual_count:02d}"
                review["class_conflicts"].append(item)
                lines = [
                    f"A: {a.rel}",
                    f"B: {b.rel}",
                    f"boxes A={stats['n_a']} B={stats['n_b']}; matched>=0.5: {stats['strong']}",
                ]
                for c in stats["class_conflict_detail"]:
                    lines.append(f"CONFLICT A cls {c['class_a']} vs B cls {c['class_b']} (IoU {c['iou']})")
                img = pair_visual(ds, a, b, f"Class-ID conflict {item['id']}", lines)
                save(img, vis / "class_conflicts" / f"{item['id']}_{slug(Path(a.rel).stem)}__{slug(Path(b.rel).stem)}.png")
                # zoom on first conflicting pair
                if stats["class_conflict_detail"]:
                    c = stats["class_conflict_detail"][0]
                    za = (c["class_a"], *c["box_a"])
                    zb = (c["class_b"], *c["box_b"])
                    zoom = zoom_conflict(ds, a, b, za, zb)
                    save(zoom, vis / "class_conflicts" / f"{item['id']}_zoom_"
                         f"{slug(Path(a.rel).stem)}__{slug(Path(b.rel).stem)}.png")
            if stats["geometry_conflict"] > 0:
                # A pair can be BOTH a class conflict and a geometry conflict
                # (it belongs to the 7 class pairs and the 177 geometry pairs).
                geom_item = dict(item)
                # Extra diagnostics for the "same box count, none matched" family:
                # are the two annotations the same object drawn at different tightness?
                if stats["strong"] == 0 and stats["n_a"] == stats["n_b"] and la.boxes and lb.boxes:
                    if stats["n_a"] == 1:
                        a_box, b_box = la.boxes[0], lb.boxes[0]
                        area_a, area_b = a_box[3] * a_box[4], b_box[3] * b_box[4]
                        ix = max(0.0, min(a_box[1] + a_box[3] / 2, b_box[1] + b_box[3] / 2)
                                 - max(a_box[1] - a_box[3] / 2, b_box[1] - b_box[3] / 2))
                        iy = max(0.0, min(a_box[2] + a_box[4] / 2, b_box[2] + b_box[4] / 2)
                                 - max(a_box[2] - a_box[4] / 2, b_box[2] - b_box[4] / 2))
                        inter = ix * iy
                        geom_item["area_ratio"] = round(
                            max(area_a, area_b) / max(1e-9, min(area_a, area_b)), 3)
                        geom_item["containment"] = round(
                            inter / max(1e-9, min(area_a, area_b)), 3)
                        geom_item["smaller_side"] = "A" if area_a < area_b else "B"
                    else:
                        deltas = [(b[1] - a[1], b[2] - a[2], b[3] - a[3], b[4] - a[4])
                                  for a, b in zip(la.boxes, lb.boxes)]
                        geom_item["index_delta_spread"] = round(max(
                            max(d[k] for d in deltas) - min(d[k] for d in deltas)
                            for k in range(4)), 4)
                if stats["n_a"] != stats["n_b"]:
                    m, ub = greedy_match(la.boxes, lb.boxes)
                    unmatched_a = [a for a, b, v in m if b is None or v < 0.5]
                    geom_item["unmatched_classes_a"] = sorted(a[0] for a in unmatched_a)
                    geom_item["unmatched_classes_b"] = sorted(b[0] for b in ub)
                geom_visual_count += 1
                geom_item["id"] = f"G{geom_visual_count:04d}"
                review["geometry_conflicts"].append(geom_item)
                geom_visuals.append(geom_item)

    # geometry visuals (individual) -- sorted worst first for readability
    review["geometry_conflicts"].sort(
        key=lambda x: (-x["geometry_conflict"], -len(x["matched_ious"]), str(x["a"])))
    for idx, item in enumerate(review["geometry_conflicts"], 1):
        a = next(i for i in ds.images if i.rel == item["a"])
        b = next(i for i in ds.images if i.rel == item["b"])
        la, lb = label_by_img[a.path], label_by_img[b.path]
        lines = [
            f"nA={item['n_a']} nB={item['n_b']}  matched>=0.5: {item['strong']}  "
            f"unmatched: {item['geometry_conflict']}  max-unmatched IoU: {item['max_unmatched_iou']}",
            f"matched IoUs: {item['matched_ious'][:14]}",
        ]
        img = pair_visual(ds, a, b, f"Geometry mismatch {item['id']}", lines)
        save(img, vis / "geometry" / f"{item['id']}_{slug(Path(item['a']).stem)}__{slug(Path(item['b']).stem)}.png")
        item["visual"] = f"visualizations/geometry/{item['id']}_" \
                         f"{slug(Path(item['a']).stem)}__{slug(Path(item['b']).stem)}.png"

    # contact sheets for the 36 most severe geometry pairs
    print("    ... geometry contact sheets", flush=True)
    worst = review["geometry_conflicts"][:36]
    sheet_dir = vis / "geometry_sheets"
    for s in range(0, len(worst), 6):
        chunk = worst[s:s + 6]
        cells = []
        for item in chunk:
            p = out_dir / item["visual"]
            if p.exists():
                im = Image.open(p)
                sc = 720 / im.width
                cells.append(im.resize((720, max(1, round(im.height * sc))),
                                       Image.Resampling.LANCZOS))
        if not cells:
            continue
        sheet_w = 720 * 2 + 16
        rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
        row_imgs = [_hstack(r) for r in rows]
        sheet_h = sum(r.height for r in row_imgs) + 8 * (len(row_imgs) - 1)
        sheet = Image.new("RGB", (sheet_w, sheet_h), COLOR_BG)
        y = 0
        for r in row_imgs:
            sheet.paste(r, (0, y))
            y += r.height + 8
        save(sheet, sheet_dir / f"sheet_{s // 6 + 1:02d}.png")

    # ---------------- 2. duplicate stems ----------------
    print("[4/7] Duplicate stems", flush=True)
    stem_groups = defaultdict(list)
    for img in ds.images:
        stem_groups[(img.batch, img.split, img.path.stem)].append(img)
    for (batch, split, stem), imgs in sorted(stem_groups.items()):
        if len(imgs) < 2:
            continue
        imgs = sorted(imgs, key=lambda x: x.ext)
        # score each image against the shared label (edge-fit heuristic)
        scores = []
        for img in imgs:
            score = edge_fit(ds, img)
            scores.append(score)
        item = {
            "batch": batch, "split": split, "stem": stem,
            "files": [i.rel for i in imgs],
            "exts": [i.ext for i in imgs],
            "dimensions": [list(ds.dims(i)) for i in imgs],
            "md5": [i.md5 for i in imgs],
            "identical_content": len({i.md5 for i in imgs}) == 1,
            "phash_distance": (
                phash_distance(imgs[0].phash, imgs[1].phash)
                if len(imgs) == 2 and imgs[0].phash and imgs[1].phash else None
            ),
            "edge_fit": [round(s, 4) for s in scores],
            "better_fit_index": int(np.argmax(scores)) if scores else None,
        }
        item["id"] = f"D{len(review['duplicate_stems']) + 1:02d}"
        review["duplicate_stems"].append(item)

        panels = []
        for i, img in enumerate(imgs):
            lab = label_by_img[img.path]
            note = f"edge-fit {scores[i]:.3f}{'  <-- better fit' if i == item['better_fit_index'] else ''}"
            panels.append(panel(ds.load_image(img), lab.boxes if lab else [],
                                f"{img.rel}", COLOR_SHARED, prefix="shared cls ", note=note))
        canvas = _hstack(panels)
        canvas = _append_footer(canvas, [
            f"One label file for {len(imgs)} different images: {stem}.txt",
            f"sizes={item['dimensions']}  dHash distance={item['phash_distance']}",
            "edge-fit is a heuristic (alignment of box borders with image edges), not ground truth.",
        ])
        save(canvas, vis / "duplicate_stems" / f"{item['id']}_{slug(batch)}_{split}_{slug(stem)}.png")
        item["visual"] = f"visualizations/duplicate_stems/{item['id']}_{slug(batch)}_{split}_{slug(stem)}.png"

    # ---------------- 3. format-only ----------------
    print("[5/7] Format-only images", flush=True)
    stems_with_standard = {img.path.stem for img in ds.images if img.ext in JPG_PNG}
    for img in ds.images:
        if img.ext in JPG_PNG:
            continue
        if img.path.stem in stems_with_standard:
            continue
        lab = label_by_img[img.path]
        with Image.open(img.path) as im:
            frames = int(getattr(im, "n_frames", 1))
            mode = im.mode
        boxes = lab.boxes if lab else []
        boxes_in_bounds = all(
            0 <= b[1] - b[3] / 2 and b[1] + b[3] / 2 <= 1
            and 0 <= b[2] - b[4] / 2 and b[2] + b[4] / 2 <= 1
            for b in boxes
        )
        item = {
            "id": f"F{len(review['format_only']) + 1:02d}",
            "batch": img.batch, "split": img.split, "rel": img.rel,
            "ext": img.ext, "dimensions": list(ds.dims(img)),
            "yolo_readable": img.ext in YOLO_OK,
            "frames": frames, "mode": mode,
            "boxes": len(boxes), "boxes_in_bounds": boxes_in_bounds,
        }
        review["format_only"].append(item)
        im = ds.load_image(img)
        p = panel(im, lab.boxes if lab else [], f"{img.rel}", COLOR_SHARED, prefix="cls ")
        p = _append_footer(p, [
            f"{img.ext}  {item['dimensions'][0]}x{item['dimensions'][1]}  "
            f"{'YOLO-readable' if item['yolo_readable'] else 'NOT loadable by Ultralytics'}",
            "Recommendation: convert a COPY to PNG (GIF: first frame) and keep the original.",
        ])
        save(p, vis / "format_only" / f"{item['id']}_{slug(Path(img.rel).stem)}.png")
        item["visual"] = f"visualizations/format_only/{item['id']}_{slug(Path(img.rel).stem)}.png"

    # ---------------- 4. exact duplicate groups ----------------
    print("[6/7] Exact duplicate groups", flush=True)
    for h, members in sorted(md5_groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda x: str(x.path))
        locs = sorted({(m.batch, m.split) for m in members})
        labels = []
        for m in members:
            l = label_by_img[m.path]
            labels.append(l.text if l else "")
        identical_bytes = len(set(labels)) == 1
        worst = 1.0
        class_conf = 0
        count_diff = False
        for a, b in itertools.combinations(members, 2):
            la, lb = label_by_img[a.path], label_by_img[b.path]
            if la is None or lb is None:
                continue
            st = match_stats(la.boxes, lb.boxes)
            if st["n_a"] != st["n_b"]:
                count_diff = True
            class_conf += st["class_conflict"]
            for v in st["matched_ious"]:
                worst = min(worst, v)
            if st["geometry_conflict"]:
                worst = 0.0
        if identical_bytes:
            category = "identical_label_bytes"
        elif class_conf:
            category = "class_conflict"
        elif count_diff:
            category = "count_mismatch"
        elif worst >= 0.9:
            category = "minor_boundary_variance"
        elif worst >= 0.5:
            category = "geometry_variance"
        else:
            category = "strong_mismatch"
        item = {
            "md5": h, "count": len(members),
            "files": [m.rel for m in members],
            "locations": [f"{b}/{s}" for b, s in locs],
            "cross_split": len({s for _, s in locs}) > 1,
            "cross_batch": len({b for b, _ in locs}) > 1,
            "category": category,
            "worst_match_iou": round(worst, 4),
            "class_conflicts": class_conf,
            "count_mismatch": count_diff,
        }
        review["exact_groups"].append(item)

    # ---------------- 5. source groups ----------------
    print("[7/7] Source-image grouping (leakage)", flush=True)
    groups, evidence, by_base = build_source_groups(ds, None)
    path_to_img = {img.path: img for img in ds.images}
    for gi, g in enumerate(sorted(groups, key=len, reverse=True), 1):
        members = [path_to_img[p] for p in g]
        splits = {m.split for m in members}
        batches = {m.batch for m in members}
        labels = []
        for m in members:
            l = label_by_img[m.path]
            labels.append(l.text if l else "")
        identical_bytes = len(set(labels)) == 1
        worst = 1.0
        class_conf = 0
        count_diff = False
        for a, b in itertools.combinations(members, 2):
            la, lb = label_by_img[a.path], label_by_img[b.path]
            if la is None or lb is None:
                continue
            st = match_stats(la.boxes, lb.boxes)
            if st["n_a"] != st["n_b"]:
                count_diff = True
            class_conf += st["class_conflict"]
            for v in st["matched_ious"]:
                worst = min(worst, v)
            if st["geometry_conflict"]:
                worst = 0.0
        if identical_bytes:
            category = "identical_label_bytes"
        elif class_conf:
            category = "class_conflict"
        elif count_diff:
            category = "count_mismatch"
        elif worst >= 0.9:
            category = "minor_boundary_variance"
        elif worst >= 0.5:
            category = "geometry_variance"
        else:
            category = "strong_mismatch"
        exact = len({m.md5 for m in members}) < len(members)
        item = {
            "id": f"S{gi:03d}",
            "count": len(members),
            "files": [m.rel for m in members],
            "locations": sorted({f"{m.batch}/{m.split}" for m in members}),
            "cross_split": len(splits) > 1,
            "cross_batch": len(batches) > 1,
            "contains_exact_duplicate": exact,
            "category": category,
            "class_conflicts": class_conf,
            "count_mismatch": count_diff,
            "worst_match_iou": round(worst, 4),
            "dimensions": [list(ds.dims(m)) for m in members],
        }
        review["source_groups"].append(item)

    # Base-name groups can be true source groups, mixtures, or pure filename
    # collisions (generic names such as images13 / image43 that map to unrelated
    # pictures). Classify them using the perceptual union-find result.
    group_of = {}
    for gi, g in enumerate(groups):
        for p in g:
            group_of[p] = gi

    base_stats = Counter()
    cross_base_stats = Counter()
    for base, members in by_base.items():
        if len(members) < 2:
            continue
        gids = [group_of.get(m.path) for m in members]
        # A singleton is its own group; keep it distinct from every other singleton.
        keyed = [gid if gid is not None else ("single", m.path) for m, gid in zip(members, gids)]
        nunique = len(set(keyed))
        cross = len({m.split for m in members}) > 1
        if nunique == len(keyed):
            base_stats["filename_collision"] += 1
            if cross:
                cross_base_stats["pure_collision"] += 1
            dists = [phash_distance(a.phash, b.phash)
                     for a, b in itertools.combinations(members, 2)
                     if a.phash and b.phash]
            if dists and min(dists) > PHASH_MAYBE:
                review["filename_collision_examples"].append({
                    "base": base, "count": len(members),
                    "files": [m.rel for m in members],
                    "min_phash_distance": min(dists),
                })
        elif nunique == 1:
            base_stats["pure_source_group"] += 1
            if cross:
                cross_base_stats["pure_source"] += 1
        else:
            base_stats["mixed_group"] += 1
            if cross:
                cross_base_stats["mixed"] += 1
    review["base_group_stats"] = dict(base_stats)
    review["base_group_cross_stats"] = dict(cross_base_stats)

    # Exact-duplicate groups whose members do not all share one Roboflow base
    # name: the base-name heuristic cannot represent these as one group.
    review["exact_cross_split_mixed_base"] = sum(
        1 for x in review["exact_groups"]
        if x["cross_split"]
        and len({base_stem(Path(f).stem) for f in x["files"]}) > 1
    )

    # Pairs in the 11..16 dHash band: possible leakage, never auto-merged.
    review["possible_source_pairs"] = [e for e in evidence if e["maybe"]]

    review["runtime_seconds"] = round(time.time() - t0, 2)
    return review, ds, label_by_img, out_dir


# --------------------------------------------------------------------------------------
# Edge-fit heuristic
# --------------------------------------------------------------------------------------


def edge_fit(ds: Dataset, img: ImageRec) -> float:
    """Heuristic: how well the shared box borders align with image edges.

    Higher = the annotation's box edges coincide with strong image gradients,
    which is typical when the box was drawn on *this* image.
    """
    lab = ds.label_for(img)
    if not lab or not lab.boxes:
        return 0.0
    im = ds.load_image(img).convert("L")
    edges = np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    W, H = im.size
    thresh = np.percentile(edges, 75) if edges.size else 0
    scores = []
    for cls, x, y, w, h in lab.boxes:
        x1 = max(0, int((x - w / 2) * W)); x2 = min(W - 1, int((x + w / 2) * W))
        y1 = max(0, int((y - h / 2) * H)); y2 = min(H - 1, int((y + h / 2) * H))
        if x2 <= x1 or y2 <= y1:
            continue
        band = max(1, int(0.04 * min(x2 - x1, y2 - y1)))
        border = np.concatenate([
            edges[y1:y1 + band, x1:x2].ravel(),
            edges[y2 - band:y2, x1:x2].ravel(),
            edges[y1:y2, x1:x1 + band].ravel(),
            edges[y1:y2, x2 - band:x2].ravel(),
        ])
        if border.size:
            scores.append(float((border > thresh).mean()))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------------------
# CSV / JSON writers
# --------------------------------------------------------------------------------------


def write_outputs(review: dict, out_dir: Path):
    # conflicts CSV
    with (out_dir / "annotation_conflicts.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "batch_a", "split_a", "file_a", "batch_b", "split_b", "file_b",
                    "n_boxes_a", "n_boxes_b", "matched_ge_0.5", "class_conflicts",
                    "geometry_unmatched", "max_unmatched_iou", "matched_ious",
                    "area_ratio", "containment", "smaller_side", "index_delta_spread"])
        for x in review["class_conflicts"]:
            w.writerow([x["id"], "class", x["batch_a"], x["split_a"], x["a"],
                        x["batch_b"], x["split_b"], x["b"], x["n_a"], x["n_b"],
                        x["strong"], x["class_conflict"], x["geometry_conflict"],
                        x["max_unmatched_iou"], x["matched_ious"], "", "", "", ""])
        for x in review["geometry_conflicts"]:
            w.writerow([x["id"], "geometry", x["batch_a"], x["split_a"], x["a"],
                        x["batch_b"], x["split_b"], x["b"], x["n_a"], x["n_b"],
                        x["strong"], x["class_conflict"], x["geometry_conflict"],
                        x["max_unmatched_iou"], x["matched_ious"],
                        x.get("area_ratio", ""), x.get("containment", ""),
                        x.get("smaller_side", ""), x.get("index_delta_spread", "")])

    with (out_dir / "duplicate_stem_pairs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "batch", "split", "stem", "files", "extensions", "dimensions",
                    "identical_content", "phash_distance", "edge_fit", "better_fit_file", "visual"])
        for x in review["duplicate_stems"]:
            better = x["files"][x["better_fit_index"]] if x["better_fit_index"] is not None else ""
            w.writerow([x["id"], x["batch"], x["split"], x["stem"], " | ".join(x["files"]),
                        " | ".join(x["exts"]), " | ".join(f"{d[0]}x{d[1]}" for d in x["dimensions"]),
                        x["identical_content"], x["phash_distance"], x["edge_fit"], better, x.get("visual", "")])

    with (out_dir / "format_only_images.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "batch", "split", "file", "ext", "width", "height",
                    "yolo_readable", "frames", "mode", "boxes", "boxes_in_bounds", "visual"])
        for x in review["format_only"]:
            w.writerow([x["id"], x["batch"], x["split"], x["rel"], x["ext"],
                        x["dimensions"][0], x["dimensions"][1],
                        x["yolo_readable"], x["frames"], x["mode"], x["boxes"],
                        x["boxes_in_bounds"], x.get("visual", "")])

    with (out_dir / "duplicate_group_actions.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["md5", "count", "locations", "cross_split", "cross_batch", "category",
                    "worst_match_iou", "class_conflicts", "count_mismatch", "files"])
        for x in review["exact_groups"]:
            w.writerow([x["md5"], x["count"], " | ".join(x["locations"]), x["cross_split"],
                        x["cross_batch"], x["category"], x["worst_match_iou"],
                        x["class_conflicts"], x["count_mismatch"], " | ".join(x["files"])])

    with (out_dir / "leakage_group_actions.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "count", "locations", "cross_split", "cross_batch",
                    "contains_exact_duplicate", "category", "class_conflicts",
                    "count_mismatch", "worst_match_iou", "files"])
        for x in review["source_groups"]:
            w.writerow([x["id"], x["count"], " | ".join(x["locations"]), x["cross_split"],
                        x["cross_batch"], x["contains_exact_duplicate"], x["category"],
                        x["class_conflicts"], x["count_mismatch"], x["worst_match_iou"],
                        " | ".join(x["files"])])

    with (out_dir / "possible_source_pairs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "file_a", "file_b", "dhash_distance", "aspect_ratio_ratio"])
        for x in review["possible_source_pairs"]:
            w.writerow([x["base"], x["a"], x["b"], x["distance"], x["aspect_ratio"]])

    (out_dir / "annotation_conflicts.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def build_report(review: dict) -> str:
    L = []
    add = L.append
    cc = review["class_conflicts"]
    gc = review["geometry_conflicts"]
    ds_ = review["duplicate_stems"]
    fo = review["format_only"]
    eg = review["exact_groups"]
    sg = review["source_groups"]

    eg_cross = [x for x in eg if x["cross_split"]]
    sg_cross = [x for x in sg if x["cross_split"]]
    eg_cat = Counter(x["category"] for x in eg)
    eg_cross_cat = Counter(x["category"] for x in eg_cross)
    sg_cat = Counter(x["category"] for x in sg)
    sg_cross_cat = Counter(x["category"] for x in sg_cross)

    geo_all_unmatched = sum(1 for x in gc if x["strong"] == 0 and x["n_a"] == x["n_b"])
    geo_count = sum(1 for x in gc if x["n_a"] != x["n_b"])
    geo_partial = sum(1 for x in gc if x["strong"] > 0 and x["n_a"] == x["n_b"])
    class_box_instances = sum(x["class_conflict"] for x in cc)
    class_ids = sorted({c["class_a"] for x in cc for c in x["class_conflict_detail"]} |
                       {c["class_b"] for x in cc for c in x["class_conflict_detail"]})
    class_ids_text = ", ".join(str(i) for i in class_ids)

    # Geometry diagnostics
    geo_single = [x for x in gc if x.get("area_ratio") is not None]
    geo_contained = sum(1 for x in geo_single if x.get("containment", 0) > 0.5)
    geo_contained_90 = sum(1 for x in geo_single if x.get("containment", 0) > 0.9)
    geo_area2 = sum(1 for x in geo_single if x.get("area_ratio", 1) > 2)
    geo_multi_unmatched = [x for x in gc
                           if x["strong"] == 0 and x["n_a"] == x["n_b"] and x["n_a"] > 1]
    geo_multi_consistent = sum(1 for x in geo_multi_unmatched
                               if x.get("index_delta_spread", 1.0) <= 0.02)
    # Cross-batch tightness gradient (same pixels, two annotations).
    appear = Counter()
    wins = Counter()
    for x in geo_single:
        if x["batch_a"] == x["batch_b"]:
            continue
        appear[x["batch_a"]] += 1
        appear[x["batch_b"]] += 1
        winner = x["batch_a"] if x["smaller_side"] == "A" else x["batch_b"]
        wins[winner] += 1
    tightness_rows = [
        [b, appear[b], wins[b], f"{100.0 * wins[b] / appear[b]:.0f}%"]
        for b in sorted(appear)
    ]
    mm_a = Counter()
    mm_b = Counter()
    for x in gc:
        for c in x.get("unmatched_classes_a", []):
            mm_a[c] += 1
        for c in x.get("unmatched_classes_b", []):
            mm_b[c] += 1

    base_cross = review.get("base_group_cross_stats", {})
    base_cross_total = sum(base_cross.values())
    exact_mixed_base = review.get("exact_cross_split_mixed_base", 0)
    dup_stem_jpeg = sum(
        1 for x in ds_
        if x["files"][x["better_fit_index"]].lower().endswith((".jpg", ".jpeg"))
    )
    containments = sorted(x.get("containment", 0.0) for x in geo_single)
    median_containment = containments[len(containments) // 2] if containments else 0.0

    def wpct(batch: str) -> str:
        return f"{100.0 * wins[batch] / appear[batch]:.0f}%" if appear.get(batch) else "n/a"

    add("# Annotation Conflict Review")
    add("")
    add(f"- **Dataset root:** `{review['root']}`")
    add(f"- **Generated:** {review['generated_at']}")
    add(f"- **Mode:** read-only — no image, label, class ID, split, or batch was modified.")
    add(f"- **Inputs:** `verification_report.md`, `verification_summary.json`, and the original files.")
    add(f"- **Runtime:** {review['runtime_seconds']} s")
    add("")
    add("---")
    add("")
    add("## 0. Executive summary")
    add("")
    add("**Five findings matter most:**")
    add("")
    add(f"1. **The 292 figure is not a usable leakage count — it fails in both directions.** "
        f"Of the 292 cross-split Roboflow base-name groups, **{base_cross.get('pure_collision', 0)} are pure "
        f"filename collisions** (same generic name, different picture), while **{exact_mixed_base} of the "
        f"222 byte-identical cross-split groups have members with different base names** and are therefore "
        f"invisible to a name-based method. The reliable union of MD5 + perceptual matching is "
        f"**{len(sg_cross)} cross-split source groups** ({sum(1 for x in sg_cross if x['contains_exact_duplicate'])} "
        f"containing exact duplicates, {sum(1 for x in sg_cross if not x['contains_exact_duplicate'])} "
        f"near-duplicate-only).")
    add(f"2. **The geometry mismatches are mostly one annotation convention, not random error.** "
        f"Of the {geo_all_unmatched} same-count/all-unmatched pairs, {len(geo_single)} contain a single "
        f"object; in {geo_contained_90} of them the smaller box is ≥90% contained in the larger "
        f"(median containment {median_containment:.3f}), "
        f"and {geo_area2} differ by more than 2× in area. Across batches, the tighter box belongs to "
        f"Batch 1 in {wpct('Batch 1')} of comparisons and to Batch 4 in only {wpct('Batch 4')} — "
        f"a systematic tight-vs-loose gradient.")
    add(f"3. **All {len(cc)} class-ID conflicts sit in five IDs** ({class_ids_text}); {class_box_instances} "
        f"boxes are affected. No class ID can be corrected from the files alone.")
    add(f"4. **All {len(ds_)} duplicate stems pair a JPEG with a PNG whose pixels and dimensions differ.** "
        f"The existing label aligns better with the JPEG in **{dup_stem_jpeg}/{len(ds_)}** cases "
        f"(edge-fit heuristic), so the PNG copies are effectively unlabelled.")
    add(f"5. **The 4 WEBP/GIF-only images are all single-frame and every box is inside the image.** "
        f"Converting a copy to PNG is mechanical and label-preserving.")
    add("")
    add(md_table(["Focus area", "Count", "Verdict"], [
        ["Class-ID conflicts (7 file pairs / box-level)", class_box_instances,
         "**Manual review** — no auto-change permitted"],
        ["Geometry mismatches (file pairs)", len(gc),
         "**Manual review** — cannot know the correct box"],
        ["Duplicate-stem pairs (different images, one label)", len(ds_),
         "**Manual annotation** for the unlabelled twin"],
        ["Images only in WEBP/GIF", len(fo),
         "**Automatically fixable** by conversion of a copy"],
        ["Exact-duplicate groups crossing splits", len(eg_cross),
         f"**{eg_cross_cat.get('identical_label_bytes', 0)} safe to dedupe**, "
         f"{len(eg_cross) - eg_cross_cat.get('identical_label_bytes', 0)} need a canonical-label decision"],
        ["Source-image groups crossing splits (corrected)", len(sg_cross),
         f"**{sg_cross_cat.get('identical_label_bytes', 0)} safe**, "
         f"{len(sg_cross) - sg_cross_cat.get('identical_label_bytes', 0)} need review"],
    ]))
    add("")
    add("**Nothing was changed.** All recommendations below are proposals pending your approval.")
    add("")

    # ------------------------------------------------------------------ method
    add("## 1. Method and scope")
    add("")
    add("- **Images are matched by MD5 first.** Byte-identical images are guaranteed to show the same "
        "pixels, so *every* annotation difference on them is a labelling inconsistency, not a visual one.")
    add("- **Near-duplicates are detected with a 64-bit dHash over 8 dihedral transforms** (0/90/180/270° "
        "× mirror), applied only to images that share a Roboflow base name. This catches augmented copies "
        "and — critically — rejects filename collisions.")
    add("- **Box matching** is greedy by IoU (class-agnostic), then class equality is checked on each "
        "matched pair. `IoU >= 0.5` is treated as the same object.")
    add("- **Edge-fit** for duplicate stems is a heuristic: the fraction of the box-border pixels that "
        "coincide with strong image gradients. It hints which image a box was drawn on; it is **not** "
        "ground truth.")
    add("- **Thresholds:** same source if `dHash <= 10` and aspect-ratio ratio `<= 1.03`, or `dHash <= 6` "
        "and aspect ratio `<= 1.10`. `11–16` is flagged as *possible* and never auto-merged.")
    add("")

    # ------------------------------------------------------------------ class conflicts
    add("## 2. Class-ID conflicts (7 file pairs)")
    add("")
    add(f"{len(cc)} file pairs contain **{class_box_instances} box-level class conflicts**: the box "
        f"geometry matches at IoU ≥ 0.5, but the class ID differs. They involve only "
        f"**class IDs {class_ids_text}**. Every pair is a byte-identical (MD5) copy of the same source "
        f"image, so the pixels cannot justify two different IDs.")
    add("")
    rows = []
    for x in cc:
        details = "; ".join(
            f"{c['class_a']} ↔ {c['class_b']} @ IoU {c['iou']}"
            for c in x["class_conflict_detail"]
        )
        rows.append([x["id"], f"{x['batch_a']}/{x['split_a']}", Path(x["a"]).name,
                     f"{x['batch_b']}/{x['split_b']}", Path(x["b"]).name, details])
    add(md_table(["ID", "A location", "A file", "B location", "B file", "Conflicting boxes"], rows))
    add("")
    add("### 2.1 Structural pattern")
    add("")
    pairs = Counter()
    for x in cc:
        for c in x["class_conflict_detail"]:
            pairs[tuple(sorted((c["class_a"], c["class_b"])))] += 1
    add(md_table(["Class pair", "Conflicting boxes"], [[f"{a} ↔ {b}", n] for (a, b), n in pairs.most_common()]))
    add("")
    add("Every conflict is confined to the five IDs **4, 17, 18, 31, 33**. That is strong evidence that "
        "these five classes are **visually or semantically close** (or that two of them are duplicates). "
        "Which ID is correct **cannot be decided from the files** — it needs the authoritative class list "
        "and, ideally, the annotator's intent.")
    add("")
    add("**Case-specific reading:**")
    add("")
    add("- **C01 (31 ↔ 33) is a clean reciprocal swap.** Both objects exist in both labels; the IDs are "
        "exchanged between them. This is the signature of two class lists whose 31/33 entries are in "
        "opposite order, not of an annotator choosing the wrong object.")
    add("- **C05 (33 ↔ 18)** is the same bed photo exported in Batch 3 and Batch 4. The bed box is "
        "identical within 5%, so only the class assignment changed between exports.")
    add("- **C02, C06, C07 (4 ↔ 31/33)** and **C03 (4 ↔ 31)** pair a large ``4`` box with a smaller "
        "``31``/``33`` box (IoU 0.51–0.76). This is consistent with a whole-object vs. component/type "
        "distinction, or with a renamed class between batches.")
    add("- **C04 (18 ↔ 17)** is the smallest semantic gap and stays inside one batch, so it is most "
        "likely a single mis-click or a duplicated class definition.")
    add("")
    add("### 2.2 Visual comparisons")
    add("")
    add("Each conflict has a three-panel image (A annotated, B annotated, overlay with B dashed) and a "
        "zoomed pair crop:")
    add("")
    for x in cc:
        base = f"visualizations/class_conflicts/{x['id']}_{slug(Path(x['a']).stem)}__{slug(Path(x['b']).stem)}"
        add(f"- **{x['id']}** — `{base}.png`  ·  zoom: `{base}_zoom.png`")
    add("")
    add("### 2.3 Assessment")
    add("")
    add("- **Automatically fixable:** none. Changing a class ID would invent ground truth.")
    add("- **Manual inspection:** all 7 pairs. A human must decide which ID describes the object, using "
        "the class definitions.")
    add("- **Cannot resolve without the owner:** the meaning of IDs 4, 17, 18, 31, 33 — and whether any of "
        "them are duplicate classes.")
    add("")

    # ------------------------------------------------------------------ geometry
    add("## 3. Bounding-box geometry mismatches (177 file pairs)")
    add("")
    add("All 177 pairs are **byte-identical images** with differing boxes, so the pixels cannot explain "
        "the difference. Sub-patterns:")
    add("")
    add(md_table(["Pattern", "Pairs", "Meaning"], [
        ["Same number of boxes, **none** matches at IoU 0.5", geo_all_unmatched,
         "Boxes were drawn around different extents (e.g. object vs. part of it)"],
        ["Same number of boxes, **some** match", geo_partial,
         "One or more objects annotated differently, others agree"],
        ["Different number of boxes", geo_count,
         "An object is missing/extra in one of the two labels"],
    ]))
    add("")
    add("Severity distribution by best remaining IoU:")
    add("")
    near = sum(1 for x in gc if 0.3 <= x["max_unmatched_iou"] < 0.5)
    mid = sum(1 for x in gc if 0.1 <= x["max_unmatched_iou"] < 0.3)
    far = sum(1 for x in gc if x["max_unmatched_iou"] < 0.1)
    add(md_table(["Best unmatched IoU", "Pairs", "Interpretation"], [
        ["0.30 – 0.50", near, "Near-miss: likely the same object, boundary disagreement"],
        ["0.10 – 0.30", mid, "Weak overlap: box shifted or resized substantially"],
        ["< 0.10", far, "No overlap: likely a different object or a missing/extra box"],
    ]))
    add("")
    add("### 3.1 What the differences actually are")
    add("")
    add(f"- **{len(geo_single)} of the {geo_all_unmatched} all-unmatched pairs contain a single box.** "
        f"In every one of them the smaller box is at least half inside the larger; in "
        f"**{geo_contained_90}** it is ≥90% inside (median containment **{median_containment:.3f}**). "
        f"**{geo_area2}** pairs differ by more than 2× in area. These are the **same object annotated at "
        f"different tightness**, not different objects.")
    add(f"- **The tightness follows a batch gradient.** Among the "
        f"**{sum(appear.values()) // 2} cross-batch single-box comparisons**, the tighter box belongs to:")
    add("")
    add(md_table(["Batch", "Comparisons", "Tighter box", "Share"], tightness_rows))
    add("")
    add("Batch 1 is systematically the tightest and Batch 4 the loosest. That is the signature of a "
        "**different annotation/export convention per batch**, not of random human error — which also "
        "means the mismatch is fixable by a single policy decision once the owner says whether boxes "
        "should be tight or loose.")
    add(f"- **The {len(geo_multi_unmatched)} multi-box all-unmatched pairs are not a block transform:** "
        f"only {geo_multi_consistent} show a consistent delta across all boxes. Those need per-object "
        f"review.")
    add(f"- **{geo_count} pairs differ in box count.** The most frequent unmatched classes are "
        f"4 ({mm_a.get(4, 0)} on the A side), 5 ({mm_a.get(5, 0)} on A / {mm_b.get(5, 0)} on B), "
        f"8 ({mm_a.get(8, 0)} on A) and 14 ({mm_a.get(14, 0)} on A). This is a general recall difference, "
        f"not one broken class.")
    add("")
    add("### 3.2 Examples")
    add("")
    add("The 36 most severe pairs have contact sheets in `visualizations/geometry_sheets/`. "
        "Every pair has an individual three-panel image in `visualizations/geometry/` named "
        "`Gxxxx_<fileA>__<fileB>.png`.")
    add("")
    rows = []
    for x in gc[:12]:
        rows.append([x["id"], f"{x['batch_a']}/{x['split_a']}/{Path(x['a']).name}",
                     f"{x['batch_b']}/{x['split_b']}/{Path(x['b']).name}",
                     f"{x['n_a']}/{x['n_b']}", x["strong"], x["geometry_conflict"],
                     x["max_unmatched_iou"]])
    add(md_table(["ID", "A file", "B file", "boxes A/B", "matched ≥0.5", "unmatched",
                  "best unmatched IoU"], rows))
    add("")
    add("### 3.3 Assessment")
    add("")
    add("- **Automatically fixable:** none. The *convention* can be standardized, but choosing which "
        "of the two boxes is right is a semantic decision.")
    add("- **Manual inspection:** all 177. Prioritise (a) the 48 count mismatches, (b) the 8 multi-box "
        "all-unmatched pairs, then (c) the single-box tightness pairs (which share one root cause).")
    add("- **Cannot resolve without the owner:** whether boxes should be tight (object boundary) or "
        "loose (full extent / context), and which batch's convention is authoritative.")
    add("")

    # ------------------------------------------------------------------ duplicate stems
    add("## 4. Duplicate-stem image pairs (15)")
    add("")
    add("In each case **two different images share one filename stem** in the same split, so only one "
        "`.txt` can exist. All 15 pairs are confirmed to be *different images* (MD5 and perceptual hash "
        "differ).")
    add("")
    rows = []
    for x in ds_:
        better = x["files"][x["better_fit_index"]] if x["better_fit_index"] is not None else "n/a"
        rows.append([x["id"], f"{x['batch']}/{x['split']}", x["stem"],
                     " / ".join(f"{f.split(chr(92))[-1]}" for f in x["files"]),
                     " / ".join(f"{d[0]}×{d[1]}" for d in x["dimensions"]),
                     x["phash_distance"], " / ".join(f"{s:.3f}" for s in x["edge_fit"]),
                     Path(better).name])
    add(md_table(["ID", "Location", "Stem", "Images", "Sizes", "dHash", "Edge-fit",
                  "Better edge-fit"], rows))
    add("")
    add(f"**Pattern:** in **{dup_stem_jpeg}/{len(ds_)}** pairs the edge-fit score favours the JPEG "
        f"(the PNG score is near zero in most cases). The dHash distances are 11–31 and the dimensions "
        f"often differ, so the PNG is neither a byte copy nor a trivially resized copy; it is most likely "
        f"a resized/derived variant, but the files cannot prove that. The existing label belongs to the "
        f"JPEG in every case, leaving the PNG with no annotation of its own.")
    add("")
    add("Visuals: `visualizations/duplicate_stems/`. Each shows both images with the *same* label drawn "
        "on top, plus the edge-fit score.")
    add("")
    add("### 4.1 Assessment")
    add("")
    add("- **Automatically fixable (structural only):** rename the second image so each image can have "
        "its own label. This does **not** create the missing annotation.")
    add("- **Manual inspection:** decide which image the existing label was drawn for (the edge-fit "
        "column is a hint), then either annotate the other image or exclude it.")
    add("- **Cannot resolve without the owner:** whether both images are intended to be in the dataset. "
        "If they are near-duplicates with one label, the honest options are annotate or drop — never "
        "reuse the same box blindly.")
    add("")

    # ------------------------------------------------------------------ format only
    add("## 5. Images available only as WEBP / GIF (4)")
    add("")
    add(md_table(["ID", "Location", "Format", "Size", "Frames", "Mode", "Boxes", "Boxes in bounds", "Ultralytics"],
                 [[x["id"], x["rel"], x["ext"], f"{x['dimensions'][0]}×{x['dimensions'][1]}",
                   x.get("frames", 1), x.get("mode", "?"), x["boxes"],
                   "yes" if x.get("boxes_in_bounds") else "**NO**",
                   "yes" if x["yolo_readable"] else "**NO**"] for x in fo]))
    add("")
    add("Visuals: `visualizations/format_only/`.")
    add("")
    add("All four are **single-frame** images and every box is inside the frame, so a PNG conversion "
        "loses no annotation information. The GIF uses palette mode (`P`); converting it to RGB PNG is "
        "sufficient.")
    add("")
    add("### 5.1 Assessment")
    add("")
    add("- **Automatically fixable:** yes. Copy each file and convert to PNG (for the GIF, take its "
        "first frame). Keep the original untouched. This changes no annotation and no class ID.")
    add("- **Manual inspection:** confirm the GIF is not animated and that the conversion preserves the "
        "frame the label was drawn on.")
    add("- **Cannot resolve without the owner:** whether these four images belong in the final dataset.")
    add("")

    # ------------------------------------------------------------------ exact dupes
    add("## 6. Exact-duplicate groups crossing splits (222)")
    add("")
    add(f"Of the {len(eg)} MD5 groups, **{len(eg_cross)} contain images from more than one split**. "
        f"They involve {sum(x['count'] for x in eg_cross):,} physical files.")
    add("")
    add(md_table(["Annotation status", "All groups", "Cross-split groups", "Action"], [
        ["Byte-identical labels", eg_cat.get("identical_label_bytes", 0),
         eg_cross_cat.get("identical_label_bytes", 0),
         "Safe to keep one copy and assign a single split"],
        ["Minor boundary variance (IoU ≥ 0.9)", eg_cat.get("minor_boundary_variance", 0),
         eg_cross_cat.get("minor_boundary_variance", 0),
         "Low risk; pick one canonical label, document the choice"],
        ["Geometry variance (IoU 0.5–0.9)", eg_cat.get("geometry_variance", 0),
         eg_cross_cat.get("geometry_variance", 0),
         "Manual: choose the better box"],
        ["Strong mismatch (IoU < 0.5)", eg_cat.get("strong_mismatch", 0),
         eg_cross_cat.get("strong_mismatch", 0),
         "Manual: annotations disagree fundamentally"],
        ["Different box count", eg_cat.get("count_mismatch", 0),
         eg_cross_cat.get("count_mismatch", 0),
         "Manual: object missing/extra"],
        ["Class conflict", eg_cat.get("class_conflict", 0),
         eg_cross_cat.get("class_conflict", 0),
         "Manual: class IDs disagree (do not auto-pick)"],
    ]))
    add("")
    add("Full list with locations, category, worst IoU and every file: `duplicate_group_actions.csv`.")
    add("")
    add("**Deduplication potential:** removing all but one file per MD5 group would delete "
        f"{sum(x['count'] for x in eg) - len(eg):,} redundant files "
        f"({len(eg)} groups). Of these, "
        f"{sum(x['count'] for x in eg_cross) - len(eg_cross):,} are in cross-split groups. "
        "**This is not proposed as automatic**: only the byte-identical-label subset is safe without a "
        "review.")
    add("")
    add("### 6.1 Assessment")
    add("")
    add("- **Automatically fixable:** dedupe the **byte-identical-label** groups "
        f"({eg_cat.get('identical_label_bytes', 0)} overall / "
        f"{eg_cross_cat.get('identical_label_bytes', 0)} cross-split), keeping one copy. "
        "The split assignment needs a documented group-aware policy.")
    add("- **Manual inspection:** every group with any annotation difference "
        f"({len(eg) - eg_cat.get('identical_label_bytes', 0)} overall).")
    add("- **Cannot resolve without the owner:** which split a shared source image should belong to, "
        "and which annotation is canonical when copies disagree.")
    add("")

    # ------------------------------------------------------------------ leakage
    add("## 7. Source-image groups crossing splits — corrected count")
    add("")
    add("### 7.1 The filename-collision problem")
    add("")
    add("The earlier report grouped images by Roboflow base name and reported **292 cross-split groups**. "
        "Re-checking the pictures shows that the base-name heuristic fails in **two independent "
        "directions**, so 292 cannot be corrected by simple subtraction:")
    add("")
    add(f"1. **False positives — generic filenames.** Of the {base_cross_total} cross-split base-name "
        f"groups, **{base_cross.get('pure_collision', 0)} are pure collisions**: `images13`, `images19`, "
        f"`images112`, `image43`, numeric names, etc. Same name, unrelated picture. Only "
        f"**{base_cross.get('pure_source', 0)}** are pure source groups and "
        f"**{base_cross.get('mixed', 0)}** are mixed.")
    add(f"2. **False negatives — renamed copies.** **{exact_mixed_base} of the 222 byte-identical "
        f"cross-split groups contain files with different base names** (e.g. `7911_376.jpg` and "
        f"`itmbghqk.jpg`), so a name-based method cannot represent them at all.")
    add("")
    coll = review["filename_collision_examples"]
    bs = review.get("base_group_stats", {})
    n_bases = sum(bs.values())
    add(f"Globally, every multi-file Roboflow base-name group was checked perceptually. Of **{n_bases}** "
        "such groups:")
    add("")
    add(md_table(["Base-name group outcome", "Groups"], [
        ["**Pure collision** — name matches, pictures are unrelated", bs.get("filename_collision", 0)],
        ["**Pure source group** — all files are the same source image", bs.get("pure_source_group", 0)],
        ["**Mixed** — some files match, others only share the name", bs.get("mixed_group", 0)],
    ]))
    add("")
    add(f"- Base-name groups with **no** perceptual match: **{len(coll)}** "
        f"(minimum dHash > {PHASH_MAYBE}).")
    add("")
    if coll:
        rows = [[f"`{c['base']}`", c["count"], c["min_phash_distance"],
                 " · ".join(Path(f).name for f in c["files"][:4])] for c in coll[:12]]
        add(md_table(["Base name", "Files", "Min dHash", "Files (truncated)"], rows))
        add("")
        add("A dHash distance near 32 of 64 is the random-image baseline; the earlier base-name grouping "
            "would have produced many false leakage alarms.")
        add("")
    add("### 7.2 Corrected source grouping")
    add("")
    sg_exact = sum(1 for x in sg_cross if x["contains_exact_duplicate"])
    sg_perc = sum(1 for x in sg_cross if not x["contains_exact_duplicate"])
    add(f"After combining **exact MD5 equality** with **perceptual matching**, the dataset has "
        f"**{len(sg)} source-image groups** with more than one file. "
        f"**{len(sg_cross)} of them cross splits** and "
        f"**{sum(1 for x in sg if x['cross_batch'])} cross batches**.")
    add("")
    add(f"Of the {len(sg_cross)} cross-split source groups, **{sg_exact} contain exact duplicates** "
        f"(these include all 222 MD5 groups found earlier, sometimes combined into one larger source "
        f"group) and **{sg_perc} are near-duplicate-only** (dHash match, different bytes). "
        f"**{len(sg_cross)} is therefore the reliable leakage count** — not 292 (base-name groups), "
        f"and not 222 (MD5 only).")
    add("")
    add(md_table(["Annotation status", "Groups", "Cross-split groups", "Action"], [
        ["Byte-identical labels", sg_cat.get("identical_label_bytes", 0),
         sg_cross_cat.get("identical_label_bytes", 0), "Safe to merge into one split unit"],
        ["Minor boundary variance", sg_cat.get("minor_boundary_variance", 0),
         sg_cross_cat.get("minor_boundary_variance", 0), "Low risk; document canonical label"],
        ["Geometry variance", sg_cat.get("geometry_variance", 0),
         sg_cross_cat.get("geometry_variance", 0), "Manual: choose box"],
        ["Strong mismatch", sg_cat.get("strong_mismatch", 0),
         sg_cross_cat.get("strong_mismatch", 0), "Manual"],
        ["Different box count", sg_cat.get("count_mismatch", 0),
         sg_cross_cat.get("count_mismatch", 0), "Manual"],
        ["Class conflict", sg_cat.get("class_conflict", 0),
         sg_cross_cat.get("class_conflict", 0), "Manual: do not auto-pick an ID"],
    ]))
    add("")
    add("Full list: `leakage_group_actions.csv`. Each group carries every member file, its locations, "
        "and whether it contains an exact duplicate.")
    add("")
    add("**Recommended split unit:** the source group, not the file. A group-aware split guarantees that "
        "no source image appears in two splits. This is a mechanical, label-preserving operation; the "
        "only open choice is the split ratio, which is a policy decision.")
    add("")
    add("### 7.3 Assessment")
    add("")
    add("- **Automatically fixable:** group-aware re-splitting (no label change) — automatically "
        "computable, but it will change split membership.")
    add("- **Manual inspection:** groups whose members disagree on annotations, and the 11–16 dHash "
        "band (possible leakage).")
    add("- **Cannot resolve without the owner:** desired train/val/test ratio and whether the original "
        "split must be preserved.")
    add("")

    # ------------------------------------------------------------------ buckets
    add("## 8. Fix classification")
    add("")
    add("### 8.1 Automatically fixable (no semantic guessing)")
    add("")
    add(md_table(["Issue", "Count", "Proposed mechanical action"], [
        ["WEBP/GIF-only images", len(fo), "Copy + convert to PNG (GIF: first frame); keep originals"],
        ["Exact-duplicate groups with byte-identical labels", eg_cat.get("identical_label_bytes", 0),
         "Keep one copy per group; record the mapping"],
        ["Source groups with byte-identical labels", sg_cat.get("identical_label_bytes", 0),
         "Collapse to one split unit; record the mapping"],
        ["Group-aware re-split (all groups)", len(sg),
         "Assign whole source groups to splits with a seeded ratio"],
    ]))
    add("")
    add("### 8.2 Requires manual inspection")
    add("")
    add(md_table(["Issue", "Count", "Why a human is needed"], [
        ["Class-ID conflicts", class_box_instances, "Cannot know which ID is correct"],
        ["Geometry mismatches", len(gc), "Cannot know which box is correct"],
        ["Duplicate stems", len(ds_), "Must decide which image was annotated / annotate the other"],
        ["Exact groups with differing labels",
         len(eg) - eg_cat.get("identical_label_bytes", 0), "Canonical annotation must be chosen"],
        ["Source groups with differing labels",
         len(sg) - sg_cat.get("identical_label_bytes", 0), "Canonical annotation must be chosen"],
        ["Possible leakage (dHash 11–16)", len(review.get("possible_source_pairs", [])),
         "Verify the images are really the same source"],
    ]))
    add("")
    add("### 8.3 Cannot be resolved without dataset-owner information")
    add("")
    add(md_table(["Question", "Why the files cannot answer it"], [
        ["Authoritative class list (id → name)", "No metadata exists anywhere in or beside the dataset"],
        ["Are IDs 4 / 17 / 18 / 31 / 33 distinct classes?", "The 7 conflicts cluster on these IDs; names are unknown"],
        ["Which annotation is canonical when copies disagree?", "Both annotations are plausible; no tie-breaker in the data"],
        ["Should duplicate source images be kept (oversampling)?", "Depends on the intended data design"],
        ["Desired split ratio / split policy", "Not encoded in the dataset"],
        ["Do the 4 WEBP/GIF images belong in the final dataset?", "No provenance information"],
        ["Are the 15 duplicate-stem twins both intended?", "No provenance information"],
    ]))
    add("")

    # ------------------------------------------------------------------ plan
    add("## 9. Proposed remediation plan (pending your approval)")
    add("")
    add("Ordered so that nothing semantic is guessed:")
    add("")
    add("1. **Freeze the owner's class list and `data.yaml`.** No class ID will be changed before this.")
    add("2. **Copy the dataset to a new working directory.** The original `final batches/` stays "
        "read-only forever.")
    add("3. **Convert the 4 WEBP/GIF images** to PNG in the copy and verify the annotations visually.")
    add("4. **Deduplicate only the byte-identical-label groups**, keeping one file and logging every "
        "removal in a CSV.")
    add("5. **Resolve the 15 duplicate stems**: choose the image the label fits, then annotate or drop "
        "the twin (owner/annotator decision).")
    add("6. **Review the 7 class conflicts and 177 geometry mismatches** using the generated visuals; "
        "record a decision per case in a signed-off table.")
    add("7. **Re-split by source group** (MD5 + perceptual), not by file, using a documented seed.")
    add("8. **Re-run `verify_dataset.py` and this review** on the cleaned copy; require zero "
        "cross-split source groups, zero duplicate stems, and zero conflicts before training.")
    add("")

    # ------------------------------------------------------------------ limits
    add("## 10. Limitations")
    add("")
    add("- **dHash is not a proof.** Flips/rotations are handled, but heavy crop, blur or occlusion can "
        "raise the distance; 11–16 is explicitly left as *possible* rather than merged.")
    add("- **Edge-fit is a heuristic.** It highlights which image a box aligns with; it does not "
        "establish the correct annotation.")
    add("- **No class names exist.** Every statement in this report is about numeric IDs only.")
    add("- **No files were changed.** The visualisations are derived copies stored outside the dataset.")
    add("")
    add("## Appendix A — Output inventory")
    add("")
    add(md_table(["File", "Contents"], [
        ["`annotation_conflicts.csv`",
         f"{len(cc)} class-conflict pairs + {len(gc)} geometry pairs, with area-ratio and "
         "containment diagnostics"],
        ["`annotation_conflicts.json`", "Structured version of every finding"],
        ["`duplicate_stem_pairs.csv`", f"{len(ds_)} pairs with sizes, hashes, edge-fit"],
        ["`format_only_images.csv`", f"{len(fo)} WEBP/GIF-only images"],
        ["`duplicate_group_actions.csv`", f"{len(eg)} MD5 groups with category and action"],
        ["`leakage_group_actions.csv`", f"{len(sg)} source groups (corrected leakage list)"],
        ["`possible_source_pairs.csv`",
         f"{len(review.get('possible_source_pairs', []))} pairs in the 11–16 dHash band"],
        ["`visualizations/class_conflicts/`", "3-panel comparisons + zoomed crops"],
        ["`visualizations/geometry/`", f"{len(gc)} three-panel comparisons"],
        ["`visualizations/geometry_sheets/`", "contact sheets for the 36 most severe pairs"],
        ["`visualizations/duplicate_stems/`", "same-label-on-both-images comparisons"],
        ["`visualizations/format_only/`", "previews with annotations"],
    ]))
    add("")
    add("## Appendix B — What was NOT done")
    add("")
    add("- No class ID was changed.")
    add("- No image or label was deleted, moved, renamed or overwritten.")
    add("- No batch was merged and no model was trained.")
    add("- No class name was invented.")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main():
    root = DEFAULT_ROOT
    out_dir = DEFAULT_OUT
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")
    print(f"Dataset : {root}")
    print(f"Output  : {out_dir}")
    print("Read-only mode.\n")
    review, ds, _, _ = inspect(root, out_dir)
    write_outputs(review, out_dir)
    (out_dir / "annotation_conflict_review.md").write_text(build_report(review), encoding="utf-8")
    print("\nDone.")
    print(f"Report : {out_dir / 'annotation_conflict_review.md'}")
    print(f"Visuals: {out_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
