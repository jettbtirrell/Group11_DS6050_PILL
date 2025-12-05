import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder


class PillDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Our images need to be in RGB to work!
        img = Image.open(row['full_path']).convert('RGB')
        img = self.transform(img)
        label = int(row['label_id'])
        return img, label
    

def load_pill_data(csv_path="Final\\nlm_images_verified.csv"):
    
    df_nlm = pd.read_csv(csv_path)
    df_nlm["pill_id"] = df_nlm["nlmImageFileName"].str.split("_").str[0]
    le = LabelEncoder()
    df_nlm = df_nlm.rename(columns={'name': 'label'})
    df_nlm['label_id'] = le.fit_transform(df_nlm['label'])
    num_classes = len(le.classes_)
    train_tfms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1,
            saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # test_tfms deals with the test data and
    # does not include the rotation or flip transformations
    # because we want the model to deal with the actual image without any artificial
    # changes

    test_tfms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    # Get unique pill IDs

    pill_ids = df_nlm["pill_id"].unique()

    # We split into a 70/30 for train and test with random seed of 42

    train_ids, temp_ids = train_test_split(
        pill_ids, test_size=0.30, random_state=42, shuffle=True
    )

    # Now we split the test into 50/50
    # All together, this gives us a 70/15/15 split for training/test/val

    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=42, shuffle=True
    )

    # This helps us to ensure that we split at the pill level
    # This is critical so each pill identity/id is assigned to only one split

    train_df = df_nlm[df_nlm["pill_id"].isin(train_ids)]
    val_df   = df_nlm[df_nlm["pill_id"].isin(val_ids)]
    test_df  = df_nlm[df_nlm["pill_id"].isin(test_ids)]

    # Applying the transformations that we specified above

    train_dataset = PillDataset(train_df, transform=train_tfms)
    val_dataset   = PillDataset(val_df,   transform=test_tfms)
    test_dataset  = PillDataset(test_df,  transform=test_tfms)

    # Using dataloader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Here are the number of unique pills in each train/test/val

    print("Number of unique pills:")
    print("Train:", train_df["pill_id"].nunique())
    print("Val  :", val_df["pill_id"].nunique())
    print("Test :", test_df["pill_id"].nunique())

    # Number of images in each train/test/val

    print("\nNumber of images:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))

    train_classes = train_df['label'].nunique()
    val_classes   = val_df['label'].nunique()
    test_classes  = test_df['label'].nunique()

    print("Unique classes per split:")
    print("Train:", train_classes)
    print("Val  :", val_classes)
    print("Test :", test_classes)
    
    return (
        train_loader, val_loader, test_loader,
        train_df, val_df, test_df,
        le, num_classes
    )

