
# Standard packages
import json
import os
from collections import defaultdict
from PIL import Image

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F


# Controls
subset_size = 1000
batch_size = 8


# Custom PyTorch dataset so that PyTorch can work with our data
class AircraftDataset(Dataset):

  # Fill dataset
  def __init__(self, images_dir, annotations_path):
    with open(annotations_path, "r") as f:
      self.coco = json.load(f)

    self.images_dir = images_dir
    self.images = self.coco["images"]

    self.categories = [c for c in self.coco["categories"] if c.get("supercategory") != "none"] # Skip supercategory
    self.cat_to_label = {cat["id"]: i + 1 for i, cat in enumerate(self.categories)}
    self.label_to_name = {i + 1: cat["name"] for i, cat in enumerate(self.categories)}

    # Group annotations by image
    self.annotations = defaultdict(list)
    for ann in self.coco["annotations"]:
      if ann["category_id"] in self.cat_to_label:
        self.annotations[ann["image_id"]].append(ann)

  # Get "size" of dataset
  def __len__(self):
    return len(self.images)

  # Get image tensor and target for 1 entry in the dataset
  def __getitem__(self, i):

    # Get image
    info = self.images[i]
    image_id = info["id"]
    image_path = os.path.join(self.images_dir, info["file_name"])
    image = Image.open(image_path).convert("RGB")
    
    # Build target
    annotations = self.annotations.get(image_id, [])
    width, height = image.size
    boxes, labels, areas = [], [], []
    for ann in annotations:

      # COCO format -> PyTorch format
      x, y, w, h = ann["bbox"]
      x1 = max(0, min(float(x), width))
      y1 = max(0, min(float(y), height))
      x2 = max(0, min(float(x + w), width))
      y2 = max(0, min(float(y + h), height))
      if x2 <= x1 or y2 <= y1:
        continue

      boxes.append([x1, y1, x2, y2])
      labels.append(self.cat_to_label[ann["category_id"]])
      areas.append((x2 - x1) * (y2 - y1))

    target = {
      "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), # Reshape so 1 row is 1 bbox
      "labels": torch.tensor(labels, dtype=torch.int64),
      "image_id": torch.tensor(image_id),
      "area": torch.tensor(areas, dtype=torch.float32),
      "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
    }

    # Return image tensor and target
    return F.to_tensor(image), target


# Datasets
train_dataset = AircraftDataset(
  images_dir="data/MAR20/train",
  annotations_path="data/MAR20/train/_annotations.coco.json"
)

# Using validation set as its larger than the test set
# and it is uncompromised as it hasn't been used for model selection
test_dataset = AircraftDataset(
  images_dir="data/MAR20/valid",
  annotations_path="data/MAR20/valid/_annotations.coco.json"
)


# Data loaders
def collate_fn(batch):
  return tuple(zip(*batch))

train_loader = DataLoader(
  Subset(train_dataset, range(subset_size)),
  batch_size=batch_size,
  shuffle=True,
  collate_fn=collate_fn
)