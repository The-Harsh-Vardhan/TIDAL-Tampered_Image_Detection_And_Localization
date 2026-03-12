import json

with open(r'c:\D Drive\Projects\BigVision Assignment\notebooks\tamper_detection_v5.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("="*80)
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    preview = src[:120].replace('\n', ' | ')
    print(f"Cell {i:3d}: {c['cell_type']:8s} | {preview}")
