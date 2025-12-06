import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from load_nih_pills import load_pill_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

# Now we get to train the model!
# Again, most of this code comes from class and homework assignments
# When training we make sure to collect metrics on loss so that we are able to do error analysis and make training curves

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# Setting up so we can evaluate the model with the data we set aside


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# here we perform feature extraction :
preprocess = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])


@torch.no_grad()
def extract_features(img_path, feature_model):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    feats = feature_model(x).cpu().squeeze(0)
    return feats


def train_pretrained_resnet(epochs=10):
    (
        train_loader, val_loader, test_loader,
        train_df, val_df, test_df,
        df_nlm, le, num_classes
    ) = load_pill_data()

    # Pretrained ResNet18
    # Since we are doing feature extraction, we do not use the ResNet18 that we built from scratch,
    # we just use the .resnet18 function. We set the weights as "IMAGENET1K_V1" so
    # it would be pretrained on ImageNet
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # We replaced the last layer with our own number of classes then saved the model
    model.fc = nn.Linear(512, num_classes)
    model = model.to(device)
    
    # Finally, we optimized the parameters within the model
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train loop
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d}:")
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 20)

    # Build feature extractor (fc = Identity)
    feature_model = models.resnet18(weights=None)
    feature_model.fc = nn.Identity()
    feature_model = feature_model.to(device)
    feature_model.load_state_dict(model.state_dict(), strict=False)
    feature_model.eval()

    # Extract features for *every* image
    features_dict = {}
    for idx, row in df_nlm.iterrows():
        label = row["label"]
        img_path = row["full_path"]
        print("Extracting:", label)
        features_dict[label] = extract_features(img_path, feature_model)

    return (
        model,         
        feature_model, 
        features_dict,  
        df_nlm,
        le,
        num_classes,
        test_loader
    )
