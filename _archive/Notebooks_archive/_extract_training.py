import json

NB_PATH = r"c:\D Drive\Projects\BigVision Assignment\notebooks\v6-5-tampered-image-detection-localization-run-01.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 27 is the training loop - print its full output
cell = nb["cells"][27]
outputs = cell.get("outputs", [])
for o in outputs:
    if "text" in o:
        print("".join(o["text"]))
    elif "data" in o and "text/plain" in o["data"]:
        print("".join(o["data"]["text/plain"]))
