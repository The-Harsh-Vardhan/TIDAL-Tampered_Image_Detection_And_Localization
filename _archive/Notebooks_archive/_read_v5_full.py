import json

with open(r'c:\D Drive\Projects\BigVision Assignment\notebooks\tamper_detection_v5.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

# Print full contents of cells 0-12
for i in range(min(13, len(nb['cells']))):
    c = nb['cells'][i]
    src = ''.join(c['source'])
    print(f"\n{'='*80}")
    print(f"=== CELL {i} ({c['cell_type']}) ===")
    print(f"{'='*80}")
    print(src)

print("\n\n" + "#"*80)
print("# REMAINING CELLS - SUMMARY + KEY SECTIONS")
print("#"*80)

# For remaining cells, print full content for specific interesting ones
for i in range(13, len(nb['cells'])):
    c = nb['cells'][i]
    src = ''.join(c['source'])
    print(f"\n{'='*80}")
    print(f"=== CELL {i} ({c['cell_type']}) ===")
    print(f"{'='*80}")
    # Print full for CONFIG, paths, final cells
    if i >= len(nb['cells']) - 3 or 'OUTPUT_DIR' in src or 'CHECKPOINT_DIR' in src or 'CONFIG' in src[:20]:
        print(src)
    else:
        # Print first 10 lines
        lines = src.split('\n')
        for line in lines[:10]:
            print(line)
        if len(lines) > 10:
            print(f"  ... ({len(lines) - 10} more lines)")
