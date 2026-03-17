import json, re

nb_path = r"c:\D Drive\Projects\BigVision Assignment\Notebooks\vK.10.3 Image Detection and Localisation.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

count = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "markdown":
        continue
    src = "".join(cell["source"])
    if "<details>" not in src and "<details " not in src:
        continue

    new_src = src
    new_src = re.sub(r'<details[^>]*>\s*\n?', '', new_src)
    new_src = re.sub(r'<summary>[^<]*</summary>\s*\n?', '', new_src)
    new_src = re.sub(r'</details>\s*\n?', '', new_src)
    new_src = re.sub(r'\n{3,}', '\n\n', new_src)
    new_src = new_src.strip() + "\n" if new_src.strip() else new_src

    if new_src != src:
        lines = new_src.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        count += 1

json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"Expanded {count} collapsible sections")

nb2 = json.load(open(nb_path, encoding="utf-8"))
remaining = sum(1 for c in nb2["cells"] if "<details>" in "".join(c.get("source", [])))
print(f"Remaining <details> tags: {remaining}")
