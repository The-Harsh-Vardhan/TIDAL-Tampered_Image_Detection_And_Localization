import json

with open(r'c:\D Drive\Projects\BigVision Assignment\notebooks\tamper_detection_v5.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

# Print FULL contents of cells 0-12
for i in range(min(13, len(nb['cells']))):
    c = nb['cells'][i]
    src = ''.join(c['source'])
    print(f"\n{'='*80}")
    print(f"=== CELL {i} ({c['cell_type']}) ===")
    print(f"{'='*80}")
    print(src)
