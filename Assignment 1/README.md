Mango Leaf Disease Classification
Objective
This project classifies mango leaf diseases using Decision Tree and Random Forest classifiers. By identifying diseases early, this model aids in improving crop health and yield through better management practices.

Introduction
Mango trees are economically valuable crops susceptible to various leaf diseases. Timely detection is critical, and this project employs Decision Tree and Random Forest models to classify mango leaf images as either healthy or diseased. Feature extraction from images is done using the EfficientNetB0 model to enhance classification performance.

Data Preprocessing
The dataset consists of mango leaf images in multiple disease categories. Key preprocessing steps include:

Image Resizing to 128x128 pixels for consistency.
Data Augmentation using transformations like rescaling for better generalization.
Feature Extraction with EfficientNetB0 to create input vectors for the classifiers.
Model Architecture
Decision Tree Classifier: A simple model that builds a tree structure for decision-making.
Random Forest Classifier: An ensemble of decision trees that reduces overfitting and improves accuracy.
Training Process
Models were trained on features from EfficientNetB0, with Random Forest set to 200 estimators and a maximum depth of 30. Model evaluation was done using accuracy, precision, recall, F1 score, and a confusion matrix.

Results
The Random Forest classifier outperformed the Decision Tree:

Decision Tree: Accuracy = 0.495, F1 Score = 0.50 (macro avg)
Random Forest: Accuracy = 0.6425, F1 Score = 0.64 (macro avg)
Conclusion
Random Forest provided higher accuracy and generalization, demonstrating effectiveness in handling data complexities. Further enhancements could include exploring deep learning models like CNNs for improved performance.
