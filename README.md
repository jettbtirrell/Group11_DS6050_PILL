# Group11_DS6050_PILL
The repository for our DS6050 Final Project titled P.I.L.L.

This repository includes our full pipeline for the beginning of our P.I.L.L. Project. Our goal was to build and evaluate baseline models for pill image classification utilizing the NLM dataset. Within our notebook file, we load and verify the dataset, checking for missing files, and encode over 2100 unique pill classes. After defining our transforms and building a custom PyTorch class for our Dataset, we create a 70-15-15 training/validation/test split and train a ResNet18 model implemented from scratch. While this model did learn, its accuracy remained fairly low, telling us that perhaps training a deep CNN from scratch on a dataset with thousands of classes and limited images may be too tall a task.

Instead, we shifted to what proved to be a more effective approach, which was using a pretrained ResNet18 for feature extraction. With ImageNet weights and a replaced final layer (to match the number of classes) this model trained quickly and reached strong performance, with almost a 90% validation accuracy in just 10 epochs. After saving the pretrained model, we convert it into a feature extractor and generate 512-dim embeddings for every image, storing them in a dictionary. As next steps, we plan on feeding these embeddings into a small RNN classifier.

Repositiory Contents Include:

Data Folder: A folder containing another folder titled '300'. This includes all of the image data.

Milestone 1 Folder: A folder containing the milestone 1 proposal.

Milestone 2 folder: A folder containing a 'Results' folder (with our training curve graphs and metrics table), the jupyter notebook 'Group11_Milestone2_Code.ipynb' (the code for our baseline model and graphs), and Group11_checkpoint.pdf (our write up for milestone 2).
