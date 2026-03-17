import json, sys

INPUT  = "./Fakeshield Lite/vF.0.3_FakeShield_Lite_v0.3.ipynb"
OUTPUT = "./Fakeshield Lite/vF.0.4_FakeShield_Lite_v0.4.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# === CELL 0: title version bump ===
cells[0]["source"] = [
    "# vF.0.4 FakeShield-Lite v0.4 \u2014 Baseline\n",
    "# Tampered Image Detection & Localization\n",
    "\n",
    "---\n",
    "\n",
    "**Assignment:** Big Vision Internship \u2014 Tampered Image Detection & Localization \n",
    "**Model:** FakeShield-Lite (pruned from FakeShield, Xu et al., ICLR 2025) \n",
    "**Runtime:** Kaggle Notebook \u2014 GPU T4 (16 GB VRAM) \n",
    "**Dataset:** [CASIA Splicing Detection + Localization](https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization) (connected as Kaggle input) \n",
    "**Version:** vF.0.4 \u2014 Kaggle-native baseline (SAM batch-dim fix)\n",
    "\n",
    "---"
]

# === CELL 18: Fix decode_mask batch dimension ===
old_src = "".join(cells[18]["source"])

# Find the exact fragment to replace
marker = "# Get dense embeddings (no-mask default) from prompt encoder"
start_idx = old_src.find(marker)
if start_idx < 0:
    print("ERROR: Could not find marker in cell 18", file=sys.stderr)
    sys.exit(1)

end_marker = "# We use h_prompt as (B, 1, 256) sparse embedding instead"
end_idx = old_src.find(end_marker)
if end_idx < 0:
    print("ERROR: Could not find end marker in cell 18", file=sys.stderr)
    sys.exit(1)
end_idx += len(end_marker)

old_block = old_src[start_idx:end_idx]

new_block = (
    "B = image_embeddings.shape[0]\n"
    "\n"
    "        # Get dense embeddings (no-mask default) from prompt encoder.\n"
    "        # prompt_encoder with no inputs returns batch-size-1 tensors:\n"
    "        #   sparse_dummy    : (1, 0, 256)\n"
    "        #   dense_embeddings: (1, 256, 64, 64)\n"
    "        sparse_dummy, dense_embeddings = self.prompt_encoder(\n"
    "            points=None, boxes=None, masks=None,\n"
    "        )\n"
    "\n"
    "        # Expand dense embeddings to match image batch size.\n"
    "        # Without this, SAM mask_decoder gets mismatched batch dims\n"
    "        # (dense has B=1 but image_embeddings has B=actual_batch_size).\n"
    "        dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)\n"
    "\n"
    "        # Use our learned prompt as sparse embeddings"
)

new_src = old_src[:start_idx] + new_block + old_src[end_idx:]

# Convert back to list of lines
new_lines = []
for line in new_src.split("\n"):
    new_lines.append(line + "\n")
if new_lines:
    new_lines[-1] = new_lines[-1].rstrip("\n")
cells[18]["source"] = new_lines

# === CELL 40: version bump ===
old40 = "".join(cells[40]["source"])
old40 = old40.replace("vF.0.3", "vF.0.4")
old40 = old40.replace("bug-fixes", "SAM batch-dim fix")
cells[40]["source"] = [old40]

# Write output
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {OUTPUT}")

# Verify
with open(OUTPUT, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
content = json.dumps(nb2, ensure_ascii=False)
print(f"Cells     : {len(nb2['cells'])}")
print(f"expand    : {content.count('dense_embeddings.expand')}  (should be 1)")
print(f"vF.0.4    : {content.count('vF.0.4')}  (should be >= 2)")
print(f"/content/ : {content.count('/content/')}  (should be 0)")
