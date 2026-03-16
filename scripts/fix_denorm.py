"""Fix _denorm 4-channel bug and missing ela kwarg in aug preview."""
import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"c:\D Drive\Projects\BigVision Assignment"

nbs = [
    "Notebooks/vK.11.0 Image Detection and Localisation [Pretrained ResNet34].ipynb",
    "Notebooks/vK.11.0 Image Detection and Localisation.ipynb",
    "Notebooks/vK.11.1 Image Detection and Localisation.ipynb",
    "Notebooks/vK.11.2 Image Detection and Localisation.ipynb",
    "Notebooks/vK.11.3 Image Detection and Localisation.ipynb",
]

OLD_DENORM = (
    'def _denorm(img_t):\n'
    '    """Reverse ImageNet normalization for display."""\n'
    '    mean = np.array([0.485, 0.456, 0.406])\n'
    '    std  = np.array([0.229, 0.224, 0.225])\n'
    '    img = img_t.permute(1, 2, 0).numpy() * std + mean\n'
    '    return np.clip(img, 0, 1)'
)

NEW_DENORM = (
    'def _denorm(img_t):\n'
    '    """Reverse ImageNet normalization for display (handles 4-ch RGB+ELA)."""\n'
    '    if img_t.shape[0] == 4:\n'
    '        img_t = img_t[:3]  # extract RGB, drop ELA channel\n'
    '    mean = np.array([0.485, 0.456, 0.406])\n'
    '    std  = np.array([0.229, 0.224, 0.225])\n'
    '    img = img_t.permute(1, 2, 0).numpy() * std + mean\n'
    '    return np.clip(img, 0, 1)'
)

OLD_AUG = "    augmented = aug_tf(image=raw_img, mask=raw_mask)"

NEW_AUG = (
    "    ela_raw = compute_ela(cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR), quality=CONFIG['ela_quality'])\n"
    "    ela_3ch = np.stack([ela_raw, ela_raw, ela_raw], axis=-1)\n"
    "    augmented = aug_tf(image=raw_img, mask=raw_mask, ela=ela_3ch)"
)

for rel in nbs:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"SKIP (not found): {rel}")
        continue

    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "def _denorm" not in src:
            continue

        new_src = src
        if OLD_DENORM in src:
            new_src = new_src.replace(OLD_DENORM, NEW_DENORM)
        if OLD_AUG in new_src:
            new_src = new_src.replace(OLD_AUG, NEW_AUG)

        if new_src != src:
            lines = new_src.split("\n")
            nb["cells"][i]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            changed = True
            print(f"FIXED: {rel.split('/')[-1]} (cell {i})")
        else:
            print(f"ALREADY OK: {rel.split('/')[-1]} (cell {i})")
        break

    if changed:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done.")
