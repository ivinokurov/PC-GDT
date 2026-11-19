"""COCO-style export for DE-ViT training (scene-disjoint splits).
Categories: 1 canopy, 2 shadow_pitched, 3 shadow_flat (target class of the paper).
Union masks M_shadow / M_tree are referenced per image (paper's A)."""
from __future__ import annotations
import json, os

CATS = [{"id": 1, "name": "canopy"}, {"id": 2, "name": "shadow_pitched"},
        {"id": 3, "name": "shadow_flat"}]
_CAT_ID = {c["name"]: c["id"] for c in CATS}


def export_coco(manifest, out_dir, splits=None):
    """splits: dict name -> list of scene ids; default: all in 'train'."""
    splits = splits or {"train": [e["id"] for e in manifest]}
    paths = []
    for name, ids in splits.items():
        coco = {"images": [], "annotations": [], "categories": CATS}
        aid = 1
        for e in manifest:
            if e["id"] not in ids:
                continue
            iid = e["id"] + 1
            coco["images"].append({"id": iid, "file_name": e["file"],
                                   "width": e["size"][0], "height": e["size"][1],
                                   "scene_id": e["id"], "sun_az": e["scene"]["az"],
                                   "sun_el": e["scene"]["el"],
                                   "mask_shadow": e["mshadow"], "mask_tree": e["mtree"]})
            for b in e["labels"]["boxes"]:
                x0, y0, x1, y1 = b["bbox"]
                coco["annotations"].append(
                    {"id": aid, "image_id": iid, "category_id": _CAT_ID[b["cls"]],
                     "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                     "iscrowd": 0, "canopy_id": b.get("canopy_id", -1)})
                aid += 1
        p = os.path.join(out_dir, f"coco_{name}.json")
        with open(p, "w") as f:
            json.dump(coco, f, indent=2)
        paths.append(p)
    return paths