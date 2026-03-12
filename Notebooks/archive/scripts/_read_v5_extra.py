import json

with open(r'c:\D Drive\Projects\BigVision Assignment\notebooks\tamper_detection_v5.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

# Print cells 11-12 and the last 2 cells
for i in [11, 12, 58, 59, 60]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        src = ''.join(c['source'])
        print(f"\n{'='*80}")
        print(f"=== CELL {i} ({c['cell_type']}) ===")
        print(f"{'='*80}")
        print(src)
