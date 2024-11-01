Mango Leaf Disease Classification Report

Introduction
This project focuses on classifying mango leaf diseases using two machine learning algorithms: Decision Tree Classifier and Random Forest Classifier. Using a dataset of labeled mango leaf images, the task involves preprocessing, feature extraction, model training, and evaluating model performance.
The primary steps undertaken were:
Loading and preprocessing the dataset.
Exploratory Data Analysis (EDA).
Feature extraction using a pre-trained ResNet50 model.
Classification using Decision Tree and Random Forest.
Evaluation and comparison of model performance.
Exploratory Data Analysis (EDA)
Dataset Structure: The dataset comprises images of mango leaves categorized by various diseases. It was loaded from a directory with subfolders representing different disease classes. We standardized each image size to 128×128 pixels for consistency.
Image Preprocessing:
Resizing: Images were resized to 128×128 pixels.
Rescaling: Pixel values were normalized from [0, 255] to [0, 1] to enhance model training stability.
Data Splitting: The “ImageDataGenerator” class was used to augment the data and split it into training (80%) and validation (20%) sets for effective model evaluation. The “flow_from_directory” method loaded images and applied the rescaling transformation.
Data Visualization: To better understand the dataset, a grid of 9 sample images from the training set was plotted. This visual confirmed the differences in appearance across disease classes, helping to contextualize the machine-learning task.
Feature Extraction
A pre-trained ResNet50 model was used as a feature extractor:
Architecture: The ResNet50 model, pre-trained on ImageNet, was modified to output flattened feature vectors.
Purpose: Instead of training a CNN from scratch, ResNet50 provided robust features from leaf images, which were then used as inputs to the classifiers.
Feature Extraction Process:
Images from the training and validation sets were passed through the ResNet50 model to obtain feature embeddings.
The features were stored as “x_train” and “x_val”, while labels were stored in “y_train” and “y_val”.
Model Training
Two classifiers, a Decision Tree and a Random Forest, were trained on the extracted features. Both classifiers are suitable for tabular data and are particularly effective when combined with feature embeddings from a CNN.
Decision Tree Classifier
Algorithm: A single-tree model that makes binary decisions at each node.
Training: The model was trained on “x_train” with corresponding “y_train” labels.
Random Forest Classifier
Algorithm: An ensemble of 100 decision trees, aggregating predictions to improve generalization and reduce overfitting.
Training: Similar to Decision Tree, but the ensemble nature increased its robustness.
Model Evaluation and Comparison
To assess model performance, both models were evaluated using standard classification metrics on the validation set (x_val and y_val).
Evaluation Metrics
The key evaluation metrics include:
Accuracy: Proportion of correct predictions.
Precision: Correctly predicted positive samples divided by the total predicted positive samples.
Recall: Correctly predicted positive samples divided by all samples that should have been predicted positive.
F1 Score: Harmonic mean of precision and recall, giving a balanced measure.
Evaluation Functions: The “evaluate_model” function printed each model's performance:
Accuracy Score
Classification Report: Detailed per-class precision, recall, and F1-score.
Confusion Matrix: Compared true and predicted labels to highlight the model's confusion across classes.
Model Performance Summary,
Model
Accuracy
Precision
Recall
F1 Score
Decision Tree
0.4113
0.43
0.41
0.41
Random Forest
0.6188
0.64
0.62
0.62

Accuracy: Random Forest achieved higher accuracy than Decision Tree, suggesting better generalization on the validation set.
Precision, Recall, and F1 Score: Random Forest outperformed Decision Tree across all metrics, indicating its robustness in distinguishing among disease classes.
Confusion Matrix Analysis
The confusion matrices for both models revealed patterns of misclassification:
Decision Tree: Misclassified some instances due to overfitting on specific classes.
Random Forest: Showed fewer misclassifications, demonstrating improved reliability.
Summary of Model Comparison
The model comparison chart provides a visual representation of each metric (accuracy, precision, recall, and F1 score) for both classifiers. Random Forest consistently scored higher than Decision Tree, making it the preferred model for this task.
Visual Comparison of Model Performance
A bar plot was generated to compare the performance metrics of both models:
Visualization: Each metric (accuracy, precision, recall, F1 score) was displayed in a grouped bar format.
Interpretation: The chart confirms that Random Forest outperforms Decision Tree in every metric, making it the recommended model for mango leaf disease classification.
The plot was customized with:
Title: "Model Performance Comparison"
Labels: Detailed axis labels and class names.
Color Palette: Improved readability with contrasting colors.
Data Labels: Each bar was labeled with the respective metric score for clarity.
Conclusion
This project successfully implemented and evaluated two classifiers for mango leaf disease classification, achieving the following key outcomes:
EDA Findings: Visualized and prepared image data effectively, gaining insights into the dataset structure.
Model Performance: Random Forest performed best in terms of accuracy, precision, recall, and F1 score.
Model Selection: Based on the results, Random Forest is recommended for this task due to its superior performance.


