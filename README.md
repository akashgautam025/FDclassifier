# Food Classifier
 This repository contains scripts and resources used to train a machine learning model for classifying various types of food images.

## Project Structure

The project has the following structure:

```
.
├── data/                      # The directory where you keep your dataset
│   ├── ApplePie/              # Directory containing apple pie images
│   ├── BagelSandwitch/        # Directory containing bagel sandwich images
│   ├── Bibimbop/              # Directory containing bibimbop images
│   ├── Bread/                 # Directory containing bread images
│   ├── FriedRice/             # Directory containing fried rice images
│   └── Pork/                  # Directory containing pork images
│
├── secret_test_cases/         # The directory where you keep your secret test images
│
├── models/                    # The directory where the trained model and related files will be saved
│   
│
├── dataset.py                 # Python script for loading and preprocessing the data
│
├── model.py                   # Python script for creating the model
│
├── train.py                   # Python script for training the model
│
├── main.py                    # Main Python script to run the project
│
└── inference.py               # Python script to run predictions on the secret test cases
```

## Approach

We approached this project using deep learning methods and built our model based on the MobileNetV2 architecture. The MobileNetV2 model was chosen for its efficient performance and accuracy trade-offs. It's pretrained on the ImageNet dataset, allowing us to use transfer learning for our specific problem.

## Training

Training was done in two phases: initial training and fine-tuning. 

1. **Initial Training**: During initial training, the base MobileNetV2 model was frozen and only the newly added layers were trained. This helped us to create a good initialization for the new layers.

2. **Fine-tuning**: Once the top layers started to converge, we unfroze the base MobileNetV2 model and trained the entire model end-to-end with a lower learning rate. This allowed subtle adjustments of the pretrained features and boosted the model's performance.

The training process used the Adam optimizer and categorical cross-entropy as the loss function. To tackle overfitting, we employed a dropout layer in our model and also used data augmentation techniques such as rotation, width shift, height shift, zoom, and horizontal flip during the training phase.

## Class Imbalance

Class imbalance was handled by TensorFlow's ImageDataGenerator function that performs on-the-fly data augmentation. This helped to increase the number of samples for under-represented classes.

## Evaluation Metrics

We used several metrics to evaluate our model, including categorical accuracy, precision, recall, and Area Under the Receiver Operating Characteristic Curve (AUC). These metrics give a comprehensive overview of the model's performance and can help identify any biases in predictions.

## Challenges

We faced a few challenges during this project, primarily related to managing the available computational resources. However, the MobileNetV2 model's efficiency proved beneficial in handling these challenges effectively.

## How to use

1. **Prepare your data**: Organize your images in the `data/` directory. Each category of food should be placed in its own directory (e.g., `ApplePie/`, `BagelSandwitch/` etc.).

2. **Set up secret test cases**: Put your secret test images in the `secret_test_cases/` directory. The model will not have access to these images during training.

3. **Run the scripts**: 

   * `dataset.py`: Load and preprocess your images for the machine learning model.

   * `model.py`: Define and compile your model architecture.

   * `train.py`: Train your model on your preprocessed images.

   * `main.py`: Run this script to perform all of the above steps.

4. **Inference**: After training the model, you can run `inference.py` to predict the categories of the images in the `secret_test_cases/` directory. There are already a few internet downloaded images which were tested on model and model predicted correct response for all of them.

5. **Model storage**: After training, the model and related files will be saved in the `models/` directory.
