import json
nb = json.load(open(r'C:\D Drive\Projects\BigVision Assignment\tamper_detection.ipynb', 'r', encoding='utf-8'))
md = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Total cells: {len(nb["cells"])}')
print(f'Markdown: {len(md)}')
print(f'Code: {len(code)}')
print()
for i, c in enumerate(nb['cells']):
    first_line = ''.join(c['source']).split('\n')[0][:70]
    print(f"  [{i+1:2d}] [{c['cell_type'][:4]}] {first_line}")
