
# Device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build model
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import train_dataset

def build_model(pretrained):
  weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
  model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
  model.roi_heads.box_predictor = FastRCNNPredictor(
    in_channels=model.roi_heads.box_predictor.cls_score.in_features,
    num_classes=len(train_dataset.label_to_name) + 1,
  )
  return model.to(device)


# Save model
import os
from datetime import datetime

def save_model(model):
  os.makedirs("models", exist_ok=True)
  save_path = f'models/{datetime.now().strftime("%Y%m%d%H%M%S")}.pth'
  torch.save(model.state_dict(), save_path)


# Filter predictions
from torchvision.ops import nms

def filterPrediction(prediction, confidence_threshold, iou_threshold):

  # Filter out weak predictions
  confidence_mask = prediction["scores"] > confidence_threshold
  boxes = prediction["boxes"][confidence_mask]
  scores = prediction["scores"][confidence_mask]
  labels = prediction["labels"][confidence_mask]

  # Filter out overlapping predictions
  intersection_mask = nms(boxes, scores, iou_threshold=iou_threshold)
  boxes = boxes[intersection_mask]
  scores = scores[intersection_mask]
  labels = labels[intersection_mask]

  return boxes, scores, labels


# Calculate IoU (intersection / union)
def calculate_iou(box1, box2):

  # Intersection
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])
  intersection = max(0, x2 - x1) * max(0, y2 - y1)

  # Union
  area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
  area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
  union = area1 + area2 - intersection # subtract intersection to avoid double counting

  return intersection / union if union > 0 else 0


# Match boxes based on IoU
def match_boxes(actual_boxes, predicted_boxes, iou_threshold=0.5):
  candidates = []

  for actual_idx, actual_box in enumerate(actual_boxes):
    for pred_idx, pred_box in enumerate(predicted_boxes):
      iou = calculate_iou(actual_box, pred_box)

      if iou >= iou_threshold:
        candidates.append((iou, actual_idx, pred_idx))

  candidates.sort(reverse=True)

  matched_actual = set()
  matched_predicted = set()
  matches = []

  for iou, actual_idx, pred_idx in candidates:
    if actual_idx in matched_actual:
      continue
    if pred_idx in matched_predicted:
      continue

    matched_actual.add(actual_idx)
    matched_predicted.add(pred_idx)
    matches.append((actual_idx, pred_idx))

  missed = [
    i for i in range(len(actual_boxes))
    if i not in matched_actual
  ]

  false_positives = [
    i for i in range(len(predicted_boxes))
    if i not in matched_predicted
  ]

  return matches, missed, false_positives