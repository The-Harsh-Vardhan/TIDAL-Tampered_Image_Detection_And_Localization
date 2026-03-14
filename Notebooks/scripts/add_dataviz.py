import json

nb_path = r"c:\D Drive\Projects\BigVision Assignment\Notebooks\vK.10.4 Image Detection and Localisation.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

# Insert after cell 31 (DataLoader construction), before cell 32 (Model Architecture)
# We'll insert 2 new cells: 1 markdown header + 1 code cell

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.4 Data Visualization\n",
        "\n",
        "Visual sanity check before training: sample images with their masks and augmentations,\n",
        "class distribution across splits, and image size statistics."
    ]
}

code_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ================== Pre-training data visualization ==================\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "def _denorm(img_t):\n",
        "    \"\"\"Reverse ImageNet normalization for display.\"\"\"\n",
        "    mean = np.array([0.485, 0.456, 0.406])\n",
        "    std  = np.array([0.229, 0.224, 0.225])\n",
        "    img = img_t.permute(1, 2, 0).numpy() * std + mean\n",
        "    return np.clip(img, 0, 1)\n",
        "\n",
        "# --- 1) Sample grid: 4 authentic + 4 tampered with masks ---\n",
        "fig, axes = plt.subplots(4, 3, figsize=(10, 13))\n",
        "fig.suptitle('Sample Images (top 2 authentic, bottom 2 tampered)', fontsize=13)\n",
        "\n",
        "auth_indices = [i for i in range(len(train_dataset)) if train_df.iloc[i]['label'] == 0][:2]\n",
        "tamp_indices = [i for i in range(len(train_dataset)) if train_df.iloc[i]['label'] == 1][:2]\n",
        "\n",
        "for row, idx in enumerate(auth_indices + tamp_indices):\n",
        "    img, mask, label = train_dataset[idx]\n",
        "    lbl_str = 'Authentic' if label == 0 else 'Tampered'\n",
        "    axes[row, 0].imshow(_denorm(img))\n",
        "    axes[row, 0].set_title(f'{lbl_str} (idx {idx})')\n",
        "    axes[row, 1].imshow(mask.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)\n",
        "    axes[row, 1].set_title('Ground Truth Mask')\n",
        "    axes[row, 2].imshow(_denorm(img))\n",
        "    axes[row, 2].imshow(mask.squeeze().numpy(), cmap='Reds', alpha=0.4)\n",
        "    axes[row, 2].set_title('Overlay')\n",
        "    for ax in axes[row]:\n",
        "        ax.axis('off')\n",
        "plt.tight_layout()\n",
        "plt.savefig(str(RESULTS_DIR / 'data_preview.png'), dpi=100, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "# --- 2) Class distribution per split ---\n",
        "fig, axes = plt.subplots(1, 3, figsize=(12, 3))\n",
        "for ax, (name, df) in zip(axes, [('Train', train_df), ('Val', val_df), ('Test', test_df)]):\n",
        "    counts = df['label'].value_counts().sort_index()\n",
        "    bars = ax.bar(['Authentic', 'Tampered'], [counts.get(0, 0), counts.get(1, 0)],\n",
        "                  color=['#2ecc71', '#e74c3c'])\n",
        "    ax.set_title(f'{name} (n={len(df)})')\n",
        "    ax.set_ylabel('Count')\n",
        "    for bar in bars:\n",
        "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,\n",
        "                str(int(bar.get_height())), ha='center', fontsize=9)\n",
        "fig.suptitle('Class Distribution Across Splits', fontsize=13)\n",
        "plt.tight_layout()\n",
        "plt.savefig(str(RESULTS_DIR / 'class_distribution.png'), dpi=100, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "# --- 3) Augmentation preview: same image with 4 random augmentations ---\n",
        "aug_idx = tamp_indices[0]  # pick first tampered sample\n",
        "fig, axes = plt.subplots(1, 5, figsize=(15, 3))\n",
        "fig.suptitle('Augmentation Preview (same tampered image, 4 random transforms)', fontsize=12)\n",
        "raw_row = train_df.iloc[aug_idx]\n",
        "raw_img = cv2.cvtColor(cv2.imread(raw_row['image_path']), cv2.COLOR_BGR2RGB)\n",
        "raw_img_resized = cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE))\n",
        "axes[0].imshow(raw_img_resized)\n",
        "axes[0].set_title('Original')\n",
        "axes[0].axis('off')\n",
        "aug_tf = get_train_transform()\n",
        "raw_mask = cv2.imread(raw_row['mask_path'], cv2.IMREAD_GRAYSCALE) if pd.notna(raw_row.get('mask_path')) else np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.uint8)\n",
        "for j in range(1, 5):\n",
        "    augmented = aug_tf(image=raw_img, mask=raw_mask)\n",
        "    aug_img = augmented['image']\n",
        "    axes[j].imshow(_denorm(aug_img))\n",
        "    axes[j].set_title(f'Aug {j}')\n",
        "    axes[j].axis('off')\n",
        "plt.tight_layout()\n",
        "plt.savefig(str(RESULTS_DIR / 'augmentation_preview.png'), dpi=100, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f'Visualizations saved to {RESULTS_DIR}/')"
    ]
}

# Insert at position 32 (pushing Model Architecture to 34)
nb["cells"].insert(32, markdown_cell)
nb["cells"].insert(33, code_cell)

# Update Table of Contents (cell 0) to include Data Visualization
toc_src = "".join(nb["cells"][0]["source"])
toc_src = toc_src.replace(
    "7. [Model Architecture]",
    "7. [Data Visualization](#64-data-visualization)\n8. [Model Architecture]"
)
# Renumber subsequent items
for old_n in range(12, 6, -1):
    toc_src = toc_src.replace(f"{old_n}. [", f"{old_n+1}. [")
lines = toc_src.split("\n")
nb["cells"][0]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"Inserted 2 cells at position 32-33. Total cells: {len(nb['cells'])}")

# Verify
nb2 = json.load(open(nb_path, encoding="utf-8"))
print(f"Cell 32: {nb2['cells'][32]['cell_type']} - {''.join(nb2['cells'][32]['source'])[:60]}")
print(f"Cell 33: {nb2['cells'][33]['cell_type']} - {''.join(nb2['cells'][33]['source'])[:60]}")
print(f"Cell 34: {nb2['cells'][34]['cell_type']} - {''.join(nb2['cells'][34]['source'])[:60]}")
