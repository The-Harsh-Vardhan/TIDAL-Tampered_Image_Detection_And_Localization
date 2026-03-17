import json

nb_path = r"c:\D Drive\Projects\BigVision Assignment\Notebooks\vK.10.4 Image Detection and Localisation.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))


def replace_in_cell(cell_idx, old, new, label=""):
    src = "".join(nb["cells"][cell_idx]["source"])
    if old not in src:
        print(f"  WARNING cell {cell_idx}: pattern not found for {label}")
        return False
    src = src.replace(old, new)
    lines = src.split("\n")
    nb["cells"][cell_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    print(f"  OK cell {cell_idx}: {label}")
    return True


# ============================================================
# 1. Fix TF32 deprecation warning (Cell 12)
# ============================================================
print("1. Fixing TF32 API...")
replace_in_cell(12,
    "    # TF32 for faster matmul on Ampere+ (does not affect determinism)\n"
    "    torch.backends.cuda.matmul.allow_tf32 = True\n"
    "    torch.backends.cudnn.allow_tf32 = True",

    "    # TF32 for faster matmul on Ampere+ (does not affect determinism)\n"
    "    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):\n"
    "        torch.backends.cuda.matmul.fp32_precision = 'tf32'\n"
    "        torch.backends.cudnn.conv.fp32_precision = 'tf32'\n"
    "    else:\n"
    "        torch.backends.cuda.matmul.allow_tf32 = True\n"
    "        torch.backends.cudnn.allow_tf32 = True",
    "TF32 new API with fallback")

# ============================================================
# 2. Add tqdm import (Cell 24)
# ============================================================
print("2. Adding tqdm import...")
replace_in_cell(24,
    "import albumentations as A",
    "from tqdm.auto import tqdm\n\nimport albumentations as A",
    "tqdm import")

# ============================================================
# 3. Fix ImageCompression args (Cell 27)
# ============================================================
print("3. Fixing ImageCompression args...")
replace_in_cell(27,
    "A.ImageCompression(quality_lower=50, quality_upper=90, p=0.25)",
    "A.ImageCompression(quality_range=(50, 90), p=0.25)",
    "ImageCompression quality_range")

# ============================================================
# 4. Fix ShiftScaleRotate -> Affine (Cell 27)
# ============================================================
print("4. Fixing ShiftScaleRotate -> Affine...")
replace_in_cell(27,
    "A.ShiftScaleRotate(\n"
    "            shift_limit=0.02,\n"
    "            scale_limit=0.1,\n"
    "            rotate_limit=10,\n"
    "            border_mode=cv2.BORDER_REFLECT_101,\n"
    "            p=0.5,\n"
    "        )",
    "A.Affine(\n"
    "            translate_percent={'x': (-0.02, 0.02), 'y': (-0.02, 0.02)},\n"
    "            scale=(0.9, 1.1),\n"
    "            rotate=(-10, 10),\n"
    "            border_mode=cv2.BORDER_REFLECT_101,\n"
    "            p=0.5,\n"
    "        )",
    "ShiftScaleRotate -> Affine")

# ============================================================
# 5. Add tqdm to train_one_epoch (Cell 44)
# ============================================================
print("5. Adding tqdm to train_one_epoch...")
replace_in_cell(44,
    "    for images, masks, labels in train_loader:",
    "    for images, masks, labels in tqdm(train_loader, desc=f'Train Ep{epoch}', leave=False):",
    "tqdm in train_one_epoch")

# ============================================================
# 6. Add tqdm to evaluate (Cell 45)
# ============================================================
print("6. Adding tqdm to evaluate...")
replace_in_cell(45,
    "    for images, masks, labels in loader:",
    "    for images, masks, labels in tqdm(loader, desc=name, leave=False):",
    "tqdm in evaluate")

# ============================================================
# 7. Update version references
# ============================================================
print("7. Updating version references...")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "vK.10.3" in src or "vk10.3" in src:
        src = src.replace("vK.10.3", "vK.10.4").replace("vk10.3", "vk10.4")
        lines = src.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        print(f"  OK cell {i}: version ref updated")

# ============================================================
# Save
# ============================================================
json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nSaved vK.10.4")

# ============================================================
# Verify
# ============================================================
nb2 = json.load(open(nb_path, encoding="utf-8"))
checks = {
    "tqdm import": any("from tqdm" in "".join(c["source"]) for c in nb2["cells"]),
    "tqdm in train": any("tqdm(train_loader" in "".join(c["source"]) for c in nb2["cells"]),
    "tqdm in eval": any("tqdm(loader" in "".join(c["source"]) for c in nb2["cells"]),
    "quality_range": any("quality_range" in "".join(c["source"]) for c in nb2["cells"]),
    "no quality_lower": not any("quality_lower" in "".join(c["source"]) for c in nb2["cells"]),
    "Affine": any("A.Affine(" in "".join(c["source"]) for c in nb2["cells"]),
    "no ShiftScaleRotate": not any("ShiftScaleRotate" in "".join(c["source"]) for c in nb2["cells"]),
    "fp32_precision": any("fp32_precision" in "".join(c["source"]) for c in nb2["cells"]),
    "vK.10.4 refs": any("vK.10.4" in "".join(c["source"]) for c in nb2["cells"]),
    "no vK.10.3 refs": not any("vK.10.3" in "".join(c["source"]) for c in nb2["cells"]),
}
print("\nVerification:")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}: {k}")
