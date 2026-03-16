"""Fix notebooks that reference undefined ELA_QUALITY in save cell.
Replace the two buggy config entries with the correct INPUT_TYPE variable."""
import json, glob

fixed = 0
for nb_path in sorted(glob.glob("**/*.ipynb", recursive=True)):
    if "Runs" in nb_path or "runner" in nb_path.lower():
        continue
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Check if this notebook defines ELA_QUALITY
    defines_eq = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "ELA_QUALITY " in src or "ELA_QUALITY=" in src:
            defines_eq = True
            break

    if defines_eq:
        continue

    modified = False
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        for li, line in enumerate(lines):
            if "'ela_quality': ELA_QUALITY" in line:
                # Replace this line: remove ela_quality, fix input_type on next line
                old_line = line
                new_line = line.replace("'ela_quality': ELA_QUALITY,", "")
                # Clean up whitespace — if line becomes just whitespace+newline, remove it
                if new_line.strip() == "":
                    lines[li] = ""
                else:
                    lines[li] = new_line
                # Fix the next line: 'input_type': 'ELA' -> 'input_type': INPUT_TYPE
                if li + 1 < len(lines) and "'input_type': 'ELA'" in lines[li + 1]:
                    lines[li + 1] = lines[li + 1].replace("'input_type': 'ELA'", "'input_type': INPUT_TYPE")
                modified = True
                print(f"FIXED: {nb_path} | cell[{ci}] line {li}")

    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        fixed += 1

print(f"\nFixed: {fixed} notebooks")
