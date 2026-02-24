
import json
import os

notebook_path = r"d:\Virtual try on\virtual-tryon\notebooks\04_agnostic_person_generator.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def get_densepose_guidance' in "".join(cell['source']):
        new_source = [
            "def get_densepose_guidance(parse_path, densepose_path):\n",
            "    \"\"\"\n",
            "    Generates a target guidance map where label 5 includes the Torso AND Arm surfaces.\n",
            "    This allows the FEM to correctly warp full sleeves even on sleeveless input persons.\n",
            "    DensePose Labels: 1, 2 = Torso; 11, 13 = L Arm; 12, 14 = R Arm.\n",
            "    \"\"\"\n",
            "    parse = np.array(Image.open(parse_path))\n",
            "    dp_segm = np.array(Image.open(densepose_path))\n",
            "    \n",
            "    # 1. Isolate the Torso + Arm region from DensePose\n",
            "    # This defines the total potential 'canvas' for an upper garment\n",
            "    # We include torso (1,2) and arm parts (11,12,13,14)\n",
            "    guidance_mask = np.isin(dp_segm, [1, 2, 11, 12, 13, 14]).astype(np.uint8)\n",
            "    \n",
            "    # 2. Smooth and refine the mask\n",
            "    kernel = np.ones((5,5), np.uint8)\n",
            "    neutral_target = cv2.morphologyEx(guidance_mask, cv2.MORPH_CLOSE, kernel, iterations=3)\n",
            "    neutral_target = cv2.dilate(neutral_target, kernel, iterations=2)\n",
            "    \n",
            "    # 3. Construct Guidance Map\n",
            "    # Start with original parsing but clear the upper garment area\n",
            "    guidance = parse.copy()\n",
            "    guidance[np.isin(parse, [5, 6, 7])] = 0\n",
            "    \n",
            "    # Fill the 'ideal' garment region with label 5 (Upper Clothes)\n",
            "    guidance[neutral_target == 1] = 5\n",
            "    \n",
            "    return Image.fromarray(guidance)"
        ]
        cell['source'] = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with sleeveless-to-sleeved support.")
