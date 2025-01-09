import warnings
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress Python warnings

def visualize_and_save_results(results, output_dir, conf_threshold=0.25):
    """Visualize bounding boxes and save the output images."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    count = 0  # Track the number of images displayed

    for result in results:
        if count >= 10:
            break  # Stop after displaying 10 images

        # Load the image
        img = cv2.imread(result.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

        # Create a title for the detections
        title = "Detected Underwater Plastics:"
        detected_classes = []

        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls.item())  # Class ID
                conf = float(box.conf.item())  # Confidence score

                # Skip boxes below the confidence threshold
                if conf < conf_threshold:
                    continue

                coords = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, coords)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                # Add label and confidence
                label = f"Class: {result.names[cls]} (Conf: {conf:.2f})"
                detected_classes.append(label)

                font_scale = 0.5
                font_thickness = 1
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

                # Draw text background
                cv2.rectangle(
                    img,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 255, 0),
                    thickness=cv2.FILLED
                )
                # Draw text
                cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        else:
            detected_classes.append("No Plastics Detected")

        # Save the image with bounding boxes
        output_path = os.path.join(output_dir, os.path.basename(result.path))
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Prepare the final title for visualization
        title += "\n" + "\n".join(detected_classes)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()

        count += 1

def save_model_metrics(metrics, save_dir):
    """Save accuracy and other metrics to a file."""
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.txt")

    # Access metrics from `results_dict`
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)']
    map50_95 = metrics.results_dict['metrics/mAP50-95(B)']
    accuracy = (precision + recall) / 2  # Example accuracy calculation

    with open(metrics_path, "w") as file:
        file.write("Model Metrics:\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"mAP@50: {map50:.4f}\n")
        file.write(f"mAP@50-95: {map50_95:.4f}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")

    print(f"Model metrics saved to {metrics_path}")

def train_and_predict():
    # Ensure CPU is used
    device = "cpu"

    # Step 1: Load YOLOv8 Pretrained Model
    model = YOLO('yolov8n.pt')  # Use yolov8n.pt for Nano model

    # Step 2: Train the Model
    print("Starting training...")
    model.train(
        data=r'C:\Users\HP\Downloads\Underwater Plastic Pollution\Dataset (underwater_plastics)\data.yaml',  # Dataset path
        epochs=10,                    # Number of epochs
        batch=16,                     # Batch size
        imgsz=640,                    # Image size
        device=device,                # Use CPU
        workers=4,                    # Optimize data loading
        lr0=0.001,                    # Learning rate
        weight_decay=0.0005,          # Apply weight decay
        augment=True                  # Enable data augmentation
    )

    # Step 3: Validate the Model
    print("Validating the model...")
    metrics = model.val()
    print("Validation Metrics:", metrics)

    # Save metrics including accuracy
    train_save_dir = r'C:\Users\HP\Downloads\Underwater Plastic Pollution\Dataset (underwater_plastics)\train'
    save_model_metrics(metrics, train_save_dir)

    # Step 4: Predict on Test Images
    test_images_path = r'C:\Users\HP\Downloads\Underwater Plastic Pollution\Dataset (underwater_plastics)\test\images'
    print("Running predictions on test images...")
    results = model.predict(
        source=test_images_path,
        conf=0.25,   # Confidence threshold
        iou=0.4,     # IoU threshold for overlapping boxes
        save=True
    )

    # Step 5: Visualize and Save Results
    output_dir = r'C:\Users\HP\Downloads\Underwater Plastic Pollution\Dataset (underwater_plastics)\output_images'
    print(f"Saving results with bounding boxes to {output_dir}...")
    visualize_and_save_results(results, output_dir, conf_threshold=0.25)

    # Step 6: Export the Model for Deployment (ONNX, TFLite, etc.)
    print("Exporting the model to ONNX format...")
    model.export(format='onnx', dynamic=True)  # Export to ONNX with dynamic input sizes
    print("Model exported to ONNX format successfully.")

# Add the main guard
if __name__ == "__main__":
    train_and_predict()
