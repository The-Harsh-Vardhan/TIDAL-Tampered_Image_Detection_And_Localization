import json

NB_PATH = r"c:\D Drive\Projects\BigVision Assignment\notebooks\v6-5-tampered-image-detection-localization-run-01.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

# Extract text outputs for key cells
for i, cell in enumerate(cells):
    outputs = cell.get("outputs", [])
    if not outputs:
        continue
    text_parts = []
    for o in outputs:
        if "text" in o:
            text_parts.extend(o["text"])
        elif "data" in o and "text/plain" in o["data"]:
            text_parts.extend(o["data"]["text/plain"])
    if text_parts:
        joined = "".join(text_parts)
        if len(joined) > 20:  # skip trivial outputs
            print(f"\n===== CELL {i} (lines {len(joined)} chars) =====")
            # Print first 3000 chars of each output
            print(joined[:3000])
            if len(joined) > 3000:
                print(f"... [truncated, total {len(joined)} chars]")
