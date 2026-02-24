import json
import os

notebook_path = r"d:\Virtual try on\virtual-tryon\notebooks\04_agnostic_person_generator.ipynb"

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        modified = False
        for line in source:
            stripped = line.strip()
            if stripped.startswith('person_img ='):
                new_source.append('    "person_img = r\\"virtual-tryon/data/viton-hd/train/image/00005_00.jpg\\"\n"')
                # Wait, the cell source in JSON is usually just the string list.
                # If I'm editing the object directly, it's just the string.
                new_source.append('person_img = r"virtual-tryon/data/viton-hd/train/image/00005_00.jpg"\n')
                modified = True
                count += 1
            elif stripped.startswith('parsing_img ='):
                new_source.append('parsing_img = r"virtual-tryon/outputs/schp/00005_00.png"\n')
                modified = True
                count += 1
            else:
                new_source.append(line)
        if modified:
            cell['source'] = new_source

# Actually, the JSON structure for 'source' is a list of lines. 
# Let me fix the logic above.

updated_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            # Search for the variable assignments
            if 'person_img =' in line:
                new_source.append('person_img = r"virtual-tryon/data/viton-hd/train/image/00005_00.jpg"\n')
                modified = True
                updated_count += 1
            elif 'parsing_img =' in line:
                new_source.append('parsing_img = r"virtual-tryon/outputs/schp/00005_00.png"\n')
                modified = True
                updated_count += 1
            else:
                new_source.append(line)
        if modified:
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook updated. Applied {updated_count} replacements.")
