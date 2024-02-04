# Detect-Anchors

This repository introduces a customizable image classification framework designed to process and classify images from the KITTI dataset, leveraging a modified ResNet model for enhanced performance. The framework is structured to accommodate various image transformations and dataset partitions, facilitating effective training and evaluation phases. The core of the project lies in the innovative use of a pre-trained ResNet model, which is fine-tuned to adapt to the specific characteristics of the KITTI dataset's images, encompassing a diverse range of urban scenes and objects.

The repository is structured around several key components:

FormatData: A custom dataset loader that preprocesses images and labels from the KITTI dataset, applying specified transformations to augment the data and improve model generalization.
ResNet: A modified version of the ResNet architecture, tailored to classify images into predefined categories with improved accuracy.
Training and Evaluation Scripts: Dedicated scripts for training the model with the KITTI dataset, implementing a variety of data augmentation techniques, and evaluating the model's performance on a separate test set.
This framework is designed with flexibility in mind, allowing users to specify dataset directories, model parameters, and training options. It demonstrates a practical application of deep learning techniques for object detection and classification in autonomous driving scenarios.
