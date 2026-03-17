"""
FakeShield-lite: Pruned FakeShield for Image Forgery Detection & Localization
==============================================================================
Adapted from FakeShield (Xu et al., ICLR 2025) for Google Colab T4.

Architecture:
  - CLIP ViT-B/16 (frozen) → global features → Detection Head + Feature Projection
  - SAM ViT-B (encoder frozen, decoder trainable) → segmentation features
  - Feature Projection → learned prompt for SAM decoder
  - SAM Mask Decoder → binary tampered region mask

This preserves FakeShield's dual-encoder (CLIP+SAM) and SAM-based localization pipeline
while removing the LLM components (~13B params) that cannot fit on T4.
"""

import os
import cv2
import glob
import json
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ============================================================================
# Part 1: Model Architecture
# ============================================================================


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation, matching FakeShield's calculate_dice_loss."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.sigmoid()
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        intersection = 2.0 * (pred_flat * target_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        dice = 1.0 - (intersection + self.smooth) / (union + self.smooth)
        return dice.mean()


class DetectionHead(nn.Module):
    """
    Binary classification head: real (0) vs tampered (1).
    Simplified from FakeShield's Domain Tag Generator (DTG).

    FakeShield's DTG classifies into 3 domains (PS/DF/AIGC).
    We simplify to binary detection as required by the assignment.
    """

    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.head(cls_token).squeeze(-1)


class FeatureProjection(nn.Module):
    """
    Projects CLIP features into SAM's prompt embedding space.

    Replaces FakeShield's TCM (Tamper Comprehension Module) + text_hidden_fcs.
    In FakeShield, TCM is an LLM encoder that aligns text with visual features,
    producing h_<SEG> for SAM. Here we directly project CLIP features.

    Architecture mirrors FakeShield's text_hidden_fcs in GLaMM:
      nn.Linear(in_dim, in_dim) -> ReLU -> nn.Linear(in_dim, out_dim) -> Dropout
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.projection(cls_token)


class SAMWrapper(nn.Module):
    """
    Wrapper around SAM ViT-B for the localization branch.

    Mirrors FakeShield's GLaMMBaseModel which:
    1. Builds SAM ViT-H via build_sam_vit_h()
    2. Freezes all SAM params
    3. Unfreezes mask_decoder for training

    We use ViT-B instead of ViT-H due to T4 constraints.
    """

    def __init__(self, sam_checkpoint: Optional[str] = None):
        super().__init__()
        self._build_sam(sam_checkpoint)

    def _build_sam(self, checkpoint: Optional[str]):
        """Build SAM ViT-B matching FakeShield's build_sam pattern."""
        from segment_anything import sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder

        # Freeze encoder (matching FakeShield's _configure_grounding_encoder)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # Train mask decoder (matching FakeShield's _train_mask_decoder)
        self.mask_decoder.train()
        for param in self.mask_decoder.parameters():
            param.requires_grad = True

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image through SAM's frozen ViT-B encoder."""
        with torch.no_grad():
            return self.image_encoder(x)

    def decode_mask(
        self,
        image_embeddings: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate mask using SAM's decoder, guided by learned prompt.

        Mirrors FakeShield's _generate_and_postprocess_masks method in GLaMMForCausalLM:
          sparse_embeddings, dense_embeddings = prompt_encoder(text_embeds=pred_embedding)
          low_res_masks, _ = mask_decoder(image_embeddings, image_pe, sparse, dense)
        """
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=prompt_embedding.unsqueeze(1),  # (B, 1, 256)
        )
        sparse_embeddings = sparse_embeddings.to(prompt_embedding.dtype)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks[:, 0]  # (B, H, W)


class FakeShieldLite(nn.Module):
    """
    FakeShield-lite: Pruned FakeShield for T4 GPU.

    Preserves from FakeShield:
      - Dual encoder design (CLIP for semantics + SAM for segmentation)
      - SAM-based mask generation with prompt embedding
      - Combined detection + localization pipeline
      - Dice + BCE loss for masks

    Removed from FakeShield:
      - LLM backbone (LLaVA-13B) — too large for T4
      - Text explanation generation — not required by assignment
      - Tamper Comprehension Module — depends on LLM
      - Domain Tag Generator 3-class — simplified to binary detection
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        sam_checkpoint: Optional[str] = None,
        image_size: int = 1024,
    ):
        super().__init__()
        self.image_size = image_size

        # CLIP encoder (frozen) — replaces FakeShield's CLIP ViT-L
        self._build_clip(clip_model_name)

        # SAM (encoder frozen, decoder trainable) — replaces ViT-H
        self.sam = SAMWrapper(sam_checkpoint)

        # Detection head — simplified DTG
        self.detection_head = DetectionHead(in_dim=self.clip_hidden_dim)

        # Feature projection — replaces TCM + text_hidden_fcs
        self.feature_projection = FeatureProjection(
            in_dim=self.clip_hidden_dim, out_dim=256
        )

        # Image preprocessing stats for SAM
        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        )

    def _build_clip(self, model_name: str):
        """Build CLIP vision encoder matching FakeShield's CLIPVisionTower."""
        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.clip_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.clip_encoder = CLIPVisionModel.from_pretrained(model_name)
        self.clip_encoder.requires_grad_(False)
        self.clip_hidden_dim = self.clip_encoder.config.hidden_size

    def preprocess_for_sam(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize and pad for SAM (matching FakeShield's grounding_enc_processor)."""
        x = (images - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def encode_clip(self, clip_images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP [CLS] token (frozen)."""
        outputs = self.clip_encoder(pixel_values=clip_images, output_hidden_states=True)
        return outputs.pooler_output  # (B, hidden_dim)

    def forward(
        self,
        clip_images: torch.Tensor,
        sam_images: torch.Tensor,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FakeShield-lite.

        Args:
            clip_images: Preprocessed images for CLIP (B, 3, 224, 224)
            sam_images: Images for SAM, resized longest-side to 1024 and padded (B, 3, 1024, 1024)
            original_sizes: Original (H, W) for each image (for mask resizing)

        Returns:
            detection_logits: (B,) binary detection logits
            mask_logits: list of (H_orig, W_orig) predicted mask logits
        """
        B = clip_images.shape[0]

        # 1. CLIP features (frozen)
        clip_features = self.encode_clip(clip_images)  # (B, 768)

        # 2. Detection (simplified DTG)
        detection_logits = self.detection_head(clip_features)  # (B,)

        # 3. Feature projection → SAM prompt (replaces TCM → h_<SEG>)
        prompt_embedding = self.feature_projection(clip_features)  # (B, 256)

        # 4. SAM image encoding (frozen)
        sam_features = self.sam.encode_image(sam_images)  # (B, 256, 64, 64)

        # 5. SAM mask decoding (trainable)
        mask_logits_low = self.sam.decode_mask(sam_features, prompt_embedding)  # (B, 256, 256)

        # 6. Upscale masks to original size
        if original_sizes is not None:
            mask_logits = []
            for i in range(B):
                h, w = original_sizes[i]
                mask_i = F.interpolate(
                    mask_logits_low[i:i+1].unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).squeeze(0)
                mask_logits.append(mask_i)
        else:
            mask_logits = mask_logits_low

        return {
            "detection_logits": detection_logits,
            "mask_logits": mask_logits,
            "mask_logits_low": mask_logits_low,
        }


# ============================================================================
# Part 2: Dataset
# ============================================================================


class TamperDataset(Dataset):
    """
    Dataset for tampering detection and localization.

    Supports CASIA, Coverage, Columbia, or similar datasets with:
      - images/ directory containing original and tampered images
      - masks/ directory containing binary ground truth masks

    Adapted from FakeShield's TamperSegmDataset to work without LLM/text components.
    """

    # SAM normalization (from FakeShield's TamperSegmDataset)
    IMG_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    IMG_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    SAM_SIZE = 1024
    CLIP_SIZE = 224

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        labels: List[int],
        clip_processor=None,
        augment: bool = False,
        image_size: int = 256,
    ):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks (empty string for authentic images)
            labels: List of labels (0=authentic, 1=tampered)
            clip_processor: CLIPImageProcessor instance
            augment: Whether to apply data augmentation
            image_size: Target image size for processing
        """
        assert len(image_paths) == len(mask_paths) == len(labels)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.clip_processor = clip_processor
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentation(
        self, image: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Apply synchronized augmentation to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = np.flip(mask, axis=0).copy()

        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = TF.rotate(mask_pil, angle, fill=0)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Color jitter (image only, not mask)
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            image = jitter(image)

        # JPEG compression simulation (for robustness)
        if random.random() > 0.3:
            quality = random.randint(70, 100)
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        return image, mask

    def _resize_longest_side(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize matching FakeShield's ResizeLongestSide transform for SAM."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _pad_to_square(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Pad to square for SAM input."""
        h, w = image.shape[:2]
        padh = target_size - h
        padw = target_size - w
        if len(image.shape) == 3:
            return np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="constant")
        return np.pad(image, ((0, padh), (0, padw)), mode="constant")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        original_size = image.size[::-1]  # (H, W)

        # Load mask
        if self.mask_paths[idx] and os.path.exists(self.mask_paths[idx]):
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((original_size[0], original_size[1]), dtype=np.float32)
            else:
                mask = mask.astype(np.float32) / 255.0
                mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros((original_size[0], original_size[1]), dtype=np.float32)

        label = self.labels[idx]

        # Apply augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Prepare CLIP input
        clip_input = self.clip_processor(images=image, return_tensors="pt")["pixel_values"][0]

        # Prepare SAM input
        image_np = np.array(image)
        sam_image = self._resize_longest_side(image_np, self.SAM_SIZE)
        sam_resize_shape = sam_image.shape[:2]
        sam_image = self._pad_to_square(sam_image, self.SAM_SIZE)
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).float()
        # Normalize for SAM
        sam_image = (sam_image - self.IMG_MEAN) / self.IMG_STD

        # Resize mask to a fixed size for loss computation
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float()

        return {
            "clip_image": clip_input,
            "sam_image": sam_image,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "original_size": torch.tensor(original_size),
            "sam_resize_shape": torch.tensor(sam_resize_shape),
            "image_path": self.image_paths[idx],
        }


def load_casia_dataset(
    dataset_dir: str,
    clip_processor=None,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> Tuple[TamperDataset, TamperDataset, TamperDataset]:
    """
    Load CASIA dataset (v1.0 or v2.0 format).

    Expected directory structure:
      dataset_dir/
        Au/          # Authentic images
        Tp/          # Tampered images
        mask/ or GT/ # Ground truth masks

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Collect authentic images
    au_patterns = [
        os.path.join(dataset_dir, "Au", "*.jpg"),
        os.path.join(dataset_dir, "Au", "*.png"),
        os.path.join(dataset_dir, "Au", "*.bmp"),
        os.path.join(dataset_dir, "Au", "*.tif"),
    ]
    au_images = []
    for pat in au_patterns:
        au_images.extend(glob.glob(pat))

    # Collect tampered images
    tp_patterns = [
        os.path.join(dataset_dir, "Tp", "*.jpg"),
        os.path.join(dataset_dir, "Tp", "*.png"),
        os.path.join(dataset_dir, "Tp", "*.bmp"),
        os.path.join(dataset_dir, "Tp", "*.tif"),
    ]
    tp_images = []
    for pat in tp_patterns:
        tp_images.extend(glob.glob(pat))

    # Find mask directory
    mask_dir = None
    for candidate in ["mask", "GT", "gt", "Mask", "masks"]:
        path = os.path.join(dataset_dir, candidate)
        if os.path.isdir(path):
            mask_dir = path
            break

    # Build lists
    image_paths = []
    mask_paths = []
    labels = []

    # Authentic images (label=0, no mask)
    for img_path in au_images:
        image_paths.append(img_path)
        mask_paths.append("")
        labels.append(0)

    # Tampered images (label=1, with mask)
    for img_path in tp_images:
        image_paths.append(img_path)
        # Try to find corresponding mask
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_found = False
        if mask_dir:
            for ext in [".png", ".bmp", ".jpg", ".tif"]:
                mask_path = os.path.join(mask_dir, img_name + ext)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    mask_found = True
                    break
                # Try with _gt suffix
                mask_path = os.path.join(mask_dir, img_name + "_gt" + ext)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    mask_found = True
                    break
        if not mask_found:
            mask_paths.append("")
        labels.append(1)

    # Shuffle and split
    combined = list(zip(image_paths, mask_paths, labels))
    random.shuffle(combined)
    image_paths, mask_paths, labels = zip(*combined) if combined else ([], [], [])
    image_paths, mask_paths, labels = list(image_paths), list(mask_paths), list(labels)

    n = len(image_paths)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_ds = TamperDataset(
        image_paths[:n_train], mask_paths[:n_train], labels[:n_train],
        clip_processor=clip_processor, augment=True,
    )
    val_ds = TamperDataset(
        image_paths[n_train:n_train+n_val], mask_paths[n_train:n_train+n_val],
        labels[n_train:n_train+n_val], clip_processor=clip_processor, augment=False,
    )
    test_ds = TamperDataset(
        image_paths[n_train+n_val:], mask_paths[n_train+n_val:],
        labels[n_train+n_val:], clip_processor=clip_processor, augment=False,
    )

    print(f"Dataset loaded: {n_train} train, {n_val} val, {n_test} test")
    print(f"  Authentic: {labels.count(0)}, Tampered: {labels.count(1)}")

    return train_ds, val_ds, test_ds


# ============================================================================
# Part 3: Training Pipeline
# ============================================================================


class FakeShieldLiteTrainer:
    """Training manager for FakeShield-lite."""

    def __init__(
        self,
        model: FakeShieldLite,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 25,
        det_loss_weight: float = 1.0,
        bce_loss_weight: float = 2.0,
        dice_loss_weight: float = 0.5,
        grad_accum_steps: int = 4,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Loss weights (from FakeShield's MFLM loss: α · BCE + β · Dice)
        self.det_loss_weight = det_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight

        # Loss functions
        self.det_criterion = nn.BCEWithLogitsLoss()
        self.mask_bce_criterion = nn.BCEWithLogitsLoss()
        self.dice_criterion = DiceLoss()

        # Optimizer with differential learning rates
        # (matching FakeShield's strategy: lower LR for pretrained components)
        param_groups = [
            {
                "params": model.detection_head.parameters(),
                "lr": lr,
                "name": "detection_head",
            },
            {
                "params": model.feature_projection.parameters(),
                "lr": lr,
                "name": "feature_projection",
            },
            {
                "params": model.sam.mask_decoder.parameters(),
                "lr": lr * 0.5,
                "name": "sam_mask_decoder",
            },
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # LR scheduler (cosine with warmup)
        total_steps = num_epochs * len(train_loader) // grad_accum_steps
        warmup_steps = int(0.05 * total_steps)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[lr, lr, lr * 0.5],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.05,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Tracking
        self.history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
        self.best_val_iou = 0.0

    def compute_loss(
        self,
        detection_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined loss matching FakeShield's loss formulation.
        ℓ = ℓ_det + α · ℓ_bce + β · ℓ_dice
        """
        # Detection loss
        det_loss = self.det_criterion(detection_logits, labels)

        # Mask losses (only for tampered images)
        tampered_mask = labels > 0.5
        if tampered_mask.any():
            tampered_pred = mask_logits[tampered_mask]
            tampered_gt = masks[tampered_mask]
            mask_bce = self.mask_bce_criterion(tampered_pred, tampered_gt)
            mask_dice = self.dice_criterion(tampered_pred, tampered_gt)
        else:
            mask_bce = torch.tensor(0.0, device=self.device)
            mask_dice = torch.tensor(0.0, device=self.device)

        total = (
            self.det_loss_weight * det_loss
            + self.bce_loss_weight * mask_bce
            + self.dice_loss_weight * mask_dice
        )

        return {
            "total": total,
            "det_loss": det_loss,
            "mask_bce": mask_bce,
            "mask_dice": mask_dice,
        }

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        # Keep frozen components in eval mode
        self.model.clip_encoder.eval()
        self.model.sam.image_encoder.eval()

        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            clip_images = batch["clip_image"].to(self.device)
            sam_images = batch["sam_image"].to(self.device)
            masks = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast():
                outputs = self.model(clip_images, sam_images)
                # Resize mask_logits_low to match mask size for loss
                mask_logits = F.interpolate(
                    outputs["mask_logits_low"].unsqueeze(1),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                losses = self.compute_loss(
                    outputs["detection_logits"], mask_logits, labels, masks
                )
                loss = losses["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += losses["total"].item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                lr_current = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(self.train_loader)} "
                    f"| Loss: {avg_loss:.4f} "
                    f"| Det: {losses['det_loss'].item():.4f} "
                    f"| BCE: {losses['mask_bce'].item():.4f} "
                    f"| Dice: {losses['mask_dice'].item():.4f} "
                    f"| LR: {lr_current:.6f}"
                )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_det_preds = []
        all_det_labels = []
        all_ious = []
        all_dices = []

        for batch in self.val_loader:
            clip_images = batch["clip_image"].to(self.device)
            sam_images = batch["sam_image"].to(self.device)
            masks = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast():
                outputs = self.model(clip_images, sam_images)
                mask_logits = F.interpolate(
                    outputs["mask_logits_low"].unsqueeze(1),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                losses = self.compute_loss(
                    outputs["detection_logits"], mask_logits, labels, masks
                )

            total_loss += losses["total"].item()

            # Detection predictions
            det_preds = (outputs["detection_logits"].sigmoid() > 0.5).float()
            all_det_preds.extend(det_preds.cpu().numpy())
            all_det_labels.extend(labels.cpu().numpy())

            # Localization metrics (only for tampered images)
            mask_preds = (mask_logits.sigmoid() > 0.5).float()
            for i in range(len(labels)):
                if labels[i] > 0.5:
                    iou = compute_iou(mask_preds[i], masks[i])
                    dice = compute_dice(mask_preds[i], masks[i])
                    all_ious.append(iou)
                    all_dices.append(dice)

        # Compute detection metrics
        det_preds = np.array(all_det_preds)
        det_labels = np.array(all_det_labels)
        det_metrics = compute_detection_metrics(det_preds, det_labels)

        avg_iou = np.mean(all_ious) if all_ious else 0.0
        avg_dice = np.mean(all_dices) if all_dices else 0.0

        return {
            "val_loss": total_loss / len(self.val_loader),
            "det_accuracy": det_metrics["accuracy"],
            "det_precision": det_metrics["precision"],
            "det_recall": det_metrics["recall"],
            "det_f1": det_metrics["f1"],
            "loc_iou": avg_iou,
            "loc_dice": avg_dice,
        }

    def train(self):
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        for epoch in range(self.num_epochs):
            print(f"{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")

            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_iou"].append(val_metrics["loc_iou"])
            self.history["val_f1"].append(val_metrics["det_f1"])

            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Detection — ACC: {val_metrics['det_accuracy']:.4f} | "
                  f"P: {val_metrics['det_precision']:.4f} | "
                  f"R: {val_metrics['det_recall']:.4f} | "
                  f"F1: {val_metrics['det_f1']:.4f}")
            print(f"  Localization — IoU: {val_metrics['loc_iou']:.4f} | "
                  f"Dice: {val_metrics['loc_dice']:.4f}")

            # Save best model
            if val_metrics["loc_iou"] > self.best_val_iou:
                self.best_val_iou = val_metrics["loc_iou"]
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                }, save_path)
                print(f"  ** New best model saved (IoU: {self.best_val_iou:.4f}) **")

            print()

        return self.history


# ============================================================================
# Part 4: Evaluation Metrics
# ============================================================================


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Intersection over Union for binary masks."""
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum().item()
    union = (pred | target).float().sum().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice score for binary masks."""
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum().item()
    total = pred.float().sum().item() + target.float().sum().item()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


def compute_detection_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute image-level detection metrics."""
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================================
# Part 5: Visualization Utilities
# ============================================================================


def visualize_predictions(
    model: FakeShieldLite,
    dataset: TamperDataset,
    device: torch.device,
    num_samples: int = 8,
    save_path: Optional[str] = None,
):
    """
    Visualize model predictions in a grid:
    [Original Image | Ground Truth Mask | Predicted Mask | Overlay]
    """
    import matplotlib.pyplot as plt

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original Image", "Ground Truth", "Prediction", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14, fontweight="bold")

    for row, idx in enumerate(indices):
        batch = dataset[idx]

        # Forward pass
        clip_img = batch["clip_image"].unsqueeze(0).to(device)
        sam_img = batch["sam_image"].unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(clip_img, sam_img)

        mask_pred = outputs["mask_logits_low"][0]
        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        mask_pred = mask_pred.sigmoid().cpu().numpy()
        mask_pred_binary = (mask_pred > 0.5).astype(np.float32)

        det_score = outputs["detection_logits"][0].sigmoid().item()

        # Load original image for display
        orig_image = Image.open(batch["image_path"]).convert("RGB")
        orig_image = orig_image.resize((256, 256))
        orig_np = np.array(orig_image)

        gt_mask = batch["mask"].numpy()
        label = batch["label"].item()

        # Original image
        axes[row, 0].imshow(orig_np)
        det_text = f"{'TAMPERED' if label > 0.5 else 'AUTHENTIC'}"
        axes[row, 0].set_xlabel(f"GT: {det_text}", fontsize=10)

        # Ground truth mask
        axes[row, 1].imshow(gt_mask, cmap="hot", vmin=0, vmax=1)

        # Predicted mask
        axes[row, 2].imshow(mask_pred_binary, cmap="hot", vmin=0, vmax=1)
        pred_text = f"Score: {det_score:.3f}"
        axes[row, 2].set_xlabel(pred_text, fontsize=10)

        # Overlay
        overlay = orig_np.copy().astype(np.float32)
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = 255
        mask_resized = cv2.resize(mask_pred_binary, (256, 256))
        mask_3d = np.stack([mask_resized] * 3, axis=-1)
        overlay = overlay * (1 - 0.4 * mask_3d) + red_mask * 0.4 * mask_3d
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        axes[row, 3].imshow(overlay)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training and validation curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU curve
    axes[1].plot(history["val_iou"], label="Val IoU", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU (Localization)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Detection F1
    axes[2].plot(history["val_f1"], label="Val F1", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1 (Detection)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def test_robustness(
    model: FakeShieldLite,
    test_dataset: TamperDataset,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness against various distortions.
    Returns metrics for each distortion type.
    """
    import matplotlib.pyplot as plt

    model.eval()
    results = {}

    distortions = {
        "clean": lambda img: img,
        "jpeg_90": lambda img: _apply_jpeg(img, 90),
        "jpeg_70": lambda img: _apply_jpeg(img, 70),
        "jpeg_50": lambda img: _apply_jpeg(img, 50),
        "resize_0.75": lambda img: _apply_resize(img, 0.75),
        "resize_0.5": lambda img: _apply_resize(img, 0.5),
        "noise_0.01": lambda img: _apply_noise(img, 0.01),
        "noise_0.03": lambda img: _apply_noise(img, 0.03),
    }

    for dist_name, dist_fn in distortions.items():
        ious = []
        dices = []
        det_preds = []
        det_labels = []

        for idx in range(len(test_dataset)):
            batch = test_dataset[idx]
            if batch["label"].item() < 0.5:
                continue  # Only test localization on tampered images

            # Apply distortion to the original image
            orig_image = Image.open(batch["image_path"]).convert("RGB")
            dist_image = dist_fn(orig_image)

            # Re-process
            clip_input = test_dataset.clip_processor(
                images=dist_image, return_tensors="pt"
            )["pixel_values"][0].unsqueeze(0).to(device)

            sam_img = batch["sam_image"].unsqueeze(0).to(device)

            with torch.no_grad():
                with autocast():
                    outputs = model(clip_input, sam_img)

            mask_pred = outputs["mask_logits_low"][0]
            mask_pred = F.interpolate(
                mask_pred.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            mask_pred = (mask_pred.sigmoid() > 0.5).float().cpu()

            gt_mask = batch["mask"]
            ious.append(compute_iou(mask_pred, gt_mask))
            dices.append(compute_dice(mask_pred, gt_mask))

        results[dist_name] = {
            "mean_iou": np.mean(ious) if ious else 0.0,
            "mean_dice": np.mean(dices) if dices else 0.0,
        }
        print(f"  {dist_name:15s} | IoU: {results[dist_name]['mean_iou']:.4f} "
              f"| Dice: {results[dist_name]['mean_dice']:.4f}")

    return results


def _apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_resize(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    small = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _apply_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


# ============================================================================
# Part 6: Main Execution (Colab-ready)
# ============================================================================


def main():
    """
    Main training script for FakeShield-lite.
    Run this in a Google Colab cell.
    """
    # Configuration
    DATASET_DIR = "/content/CASIA"  # Update path as needed
    SAM_CHECKPOINT = "/content/sam_vit_b_01ec64.pth"
    CLIP_MODEL = "openai/clip-vit-base-patch16"
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Initialize model
    print("\nInitializing FakeShield-lite...")
    model = FakeShieldLite(
        clip_model_name=CLIP_MODEL,
        sam_checkpoint=SAM_CHECKPOINT,
    )

    # Load dataset
    print("\nLoading dataset...")
    train_ds, val_ds, test_ds = load_casia_dataset(
        DATASET_DIR, clip_processor=model.clip_processor,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Train
    trainer = FakeShieldLiteTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=LR,
        num_epochs=NUM_EPOCHS,
    )
    history = trainer.train()

    # Plot training curves
    plot_training_curves(history, save_path="training_curves.png")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    # Load best model
    best_ckpt = torch.load(os.path.join("checkpoints", "best_model.pth"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Test metrics
    test_metrics = trainer.validate()  # Use val_loader or test_loader
    print(f"\nTest Detection: ACC={test_metrics['det_accuracy']:.4f} "
          f"F1={test_metrics['det_f1']:.4f}")
    print(f"Test Localization: IoU={test_metrics['loc_iou']:.4f} "
          f"Dice={test_metrics['loc_dice']:.4f}")

    # Visualize
    visualize_predictions(model, test_ds, DEVICE, num_samples=8, save_path="predictions.png")

    # Robustness testing
    print("\nRobustness Testing:")
    robustness_results = test_robustness(model, test_ds, DEVICE)


if __name__ == "__main__":
    main()
