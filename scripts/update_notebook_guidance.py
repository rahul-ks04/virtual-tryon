
import json
import os

notebook_path = r"d:\Virtual try on\virtual-tryon\notebooks\04_agnostic_person_generator.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where to insert the new logic. 
# We'll insert it after the get_agnostic_person function.
insert_index = -1
for i, cell in enumerate(nb['cells']):
    if "def get_agnostic_person" in "".join(cell['source']):
        insert_index = i + 1
        break

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Target Guidance Map Generation\n",
            "\n",
            "To achieve true garment independence, we provide a **Guidance Map** to the FEM (Flow Estimation Module).\n",
            "This map has label `5` (Upper-clothes) in a smooth, neutral torso region, so the network estimates flow based on the *pose* rather than the *original garment's silhouette*."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_target_guidance(parse_path, iterations=5):\n",
            "    \"\"\"\n",
            "    Generates a neutral guidance parsing map for the FEM.\n",
            "    \"\"\"\n",
            "    parse = np.array(Image.open(Path(parse_path)))\n",
            "    \n",
            "    # 1. Isolate garment area (Upper-clothes, Dress, Coat)\n",
            "    garment_mask = np.isin(parse, [5, 6, 7]).astype(np.uint8)\n",
            "    \n",
            "    # 2. Smooth/Morph the mask to remove 'bias' (wrinkles, specific edges)\n",
            "    kernel = np.ones((5,5), np.uint8)\n",
            "    # Closing fills small holes and smooths edges\n",
            "    neutral_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)\n",
            "    # Dilation ensures it fully covers the torso guidance area for the FEM\n",
            "    neutral_mask = cv2.dilate(neutral_mask, kernel, iterations=2)\n",
            "    \n",
            "    # 3. Create a Guidance Parsing Map\n",
            "    # We copy the original but replace the garment area with our 'neutral' one\n",
            "    guidance_parse = parse.copy()\n",
            "    # Set background for the garment area first\n",
            "    guidance_parse[np.isin(parse, [5, 6, 7])] = 0 \n",
            "    # Apply the neutral garment shape as label 5\n",
            "    guidance_parse[neutral_mask == 1] = 5\n",
            "    \n",
            "    return Image.fromarray(guidance_parse)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualization of the Guidance Logic\n",
            "person_id = \"00041_00.jpg\" # Ensure you have correct data paths below\n",
            "img_path = Path(r\"../data/viton-hd/train/image\") / person_id\n",
            "parse_path = Path(r\"../data/viton-hd/train/image-parse-v3\") / person_id.replace('.jpg', '.png')\n",
            "\n",
            "if img_path.exists() and parse_path.exists():\n",
            "    agnostic_img, agnostic_parse = get_agnostic_person(img_path, parse_path)\n",
            "    guidance_parse = get_target_guidance(parse_path)\n",
            "\n",
            "    fig, ax = plt.subplots(1, 3, figsize=(15, 7))\n",
            "    ax[0].imshow(Image.open(img_path))\n",
            "    ax[0].set_title(\"Original Image\")\n",
            "    ax[1].imshow(agnostic_img)\n",
            "    ax[1].set_title(\"Agnostic Person (Stage 3 Input)\")\n",
            "    ax[2].imshow(guidance_parse)\n",
            "    ax[2].set_title(\"Guidance Parsing (FEM/Stage 2 Input)\")\n",
            "    plt.show()\n",
            "else:\n",
            "    print(f\"Test paths not found. Please check paths: \\n{img_path}\\n{parse_path}\")"
        ]
    }
]

if insert_index != -1:
    nb['cells'] = nb['cells'][:insert_index] + new_cells + nb['cells'][insert_index:]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with guidance logic.")
