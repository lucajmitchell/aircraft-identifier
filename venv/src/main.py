
# Standard packages
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# PyTorch
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

# Evaluation tools
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# My packages
from data import train_dataset, test_dataset, train_loader
from utils import device, build_model, save_model, filterPrediction, match_boxes, calculate_iou


# Controls
do_training = False
num_epochs = 10

do_identification = True
model_path = "example/example_weights.pth"
image_path = random.choice(glob.glob("data/MAR20/test/*.jpg"))
confidence_threshold = 0.5
iou_threshold = 0.3

do_evaluation = False
eval_model_path = model_path


# Training
if do_training:
  model = build_model(pretrained=True)
  params = [p for p in model.parameters() if p.requires_grad]
  optimiser = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

  for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
      images, targets = batch

      # Add images and targets to device
      images = [img.to(device) for img in images]
      targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

      # Train
      optimiser.zero_grad()
      loss = sum(model(images, targets).values())
      loss.backward()
      optimiser.step()

      # Display progress
      print(f"epoch {epoch+1}/{num_epochs} iter {i+1}/{len(train_loader)}", flush=True)

  # Save model
  save_model(model)


# Identification
if do_identification:
  model = build_model(pretrained=False)
  model.load_state_dict(torch.load(eval_model_path, map_location=device, weights_only=True))
  model.eval()

  # Run detection and classification on image
  image = Image.open(image_path).convert("RGB")
  image_tensor = F.to_tensor(image).to(device)
  with torch.no_grad():
    prediction = model([image_tensor])[0]

  # Filter out weak predictions
  boxes, scores, labels = filterPrediction(
    prediction=prediction,
    confidence_threshold=confidence_threshold,
    iou_threshold=iou_threshold
  )
  boxes, scores, labels = boxes.cpu(), scores.cpu(), labels.cpu()

  # Display predictions
  display_labels = [
    f"{train_dataset.label_to_name[label.item()]} ({score.item():.2f})"
    for label, score in zip(labels, scores)
  ]
  annotated_image = draw_bounding_boxes(
    (F.to_tensor(image) * 255).to(torch.uint8),
    boxes,
    labels=display_labels,
    colors="red"
  )
  plt.figure(figsize=(8, 8), layout="tight")
  plt.imshow(annotated_image.permute(1, 2, 0))
  plt.axis("off")
  plt.show()


# Evaluation
if do_evaluation:
  model = build_model(pretrained=False)
  model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  model.eval()

  all_actual_labels = []
  all_predicted_labels = []
  for i in range(len(test_dataset)):
    image_tensor, annotations = test_dataset[i]

    # Get predictions
    with torch.no_grad():
      prediction = model([image_tensor.to(device)])[0]
    predicted_boxes, scores, predicted_labels = filterPrediction(
      prediction=prediction,
      confidence_threshold=confidence_threshold,
      iou_threshold=iou_threshold
    )
    predicted_boxes = predicted_boxes.tolist()
    predicted_labels = predicted_labels.tolist()

    # Get true values
    actual_boxes = annotations["boxes"].tolist()
    actual_labels = annotations["labels"].tolist()

    # Match predicted boxes to actual boxes
    matches, missed, false_positives = match_boxes(
      actual_boxes,
      predicted_boxes,
      iou_threshold=0.5
    )

    # Collect class labels from matched aircraft.
    for actual_index, predicted_index in matches:
      all_actual_labels.append(actual_labels[actual_index])
      all_predicted_labels.append(predicted_labels[predicted_index])

    print(f"image {i+1}/{len(test_dataset)} matched {len(matches)}/{len(matches) + len(missed)}", flush=True)

  # Create classification report
  class_labels = list(test_dataset.label_to_name.keys())
  class_names = list(test_dataset.label_to_name.values())
  report = classification_report(
    all_actual_labels,
    all_predicted_labels,
    labels=class_labels,
    target_names=class_names,
    digits=4,
    zero_division=0,
    output_dict=True
  )

  # Save classification report
  report_df = pd.DataFrame(report).transpose()
  report_df.to_csv(
    "classification_report.csv",
    index=True,
    index_label="aircraft"
  )

  # Create confusion matrix
  confusion_matrix = confusion_matrix(
    all_actual_labels,
    all_predicted_labels,
    labels=class_labels
  )

  # Save confusion matrix
  plt.figure(figsize=(12, 8))
  sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
  )
  plt.xlabel("Predicted class")
  plt.ylabel("Actual class")
  plt.xticks(rotation=45)
  plt.yticks(rotation=0)
  plt.savefig("confusion_matrix.png", bbox_inches="tight")