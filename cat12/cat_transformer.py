import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, models, transforms
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import difflib
import copy
import sys

test_flag = False

if len(sys.argv) == 2:
    if sys.argv[1] == '--test':
        test_flag = True
        print('Test mode')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define transforms for data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the training and testing datasets
# Read the train_list.txt file to get the image paths and labels
with open('train_list.txt', 'r') as f:
    train_list = f.read().splitlines()

train_images = []
train_labels = []

for item in train_list:
    image_path, label = item.split()
    train_images.append(image_path)
    train_labels.append(int(label))

# Create a custom dataset for training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)

# Split train_list into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
# train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Create custom datasets for training and testing
train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)

# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

class PatchEmbedding(nn.Module):
    """将图像分割为小块并进行线性嵌入"""
    def __init__(self, img_size, patch_size, in_channels, hidden_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, hidden_dim, H/P, W/P)
        x = x.flatten(2)        # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, hidden_dim)
        return x

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = hidden_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_dropout(x)
        return x

class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VisionTransformer(nn.Module):
    """ViT架构"""
    def __init__(self, img_size, patch_size, in_channels, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, dropout_rate=0.1):
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, hidden_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, hidden_dim))

        self.encoder = nn.Sequential(*[Encoder(hidden_dim, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding

        x = self.encoder(x)
        x = self.ln(x)

        return self.head(x[:, 0])

model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=1000)
model_dict = model.state_dict()

if not test_flag:

    if not os.path.exists('vit_b_16-c867db91.pth'):
        os.system('wget https://download.pytorch.org/models/vit_b_16-c867db91.pth')
    pretrained_dict = torch.load('vit_b_16-c867db91.pth')

    pretrained_dict_fix = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_fix[k] = v
        elif "conv_proj" in k:
            key = k.replace("conv_proj", "patch_embedding.projection")
            pretrained_dict_fix[key] = v
        elif "encoder.pos_embedding" in k:
            key = k.replace("encoder.pos_embedding", "position_embedding")
            pretrained_dict_fix[key] = v
        elif "ln_1" in k:
            key = k.replace("ln_1", "ln_1")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            pretrained_dict_fix[key] = v
        elif "ln_2" in k:
            key = k.replace("ln_2", "ln_2")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            pretrained_dict_fix[key] = v
        elif "self_attention.in_proj_weight" in k:
            key = k.replace("self_attention.in_proj_weight", "attention.qkv.weight")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            key = key.replace("self_attention", "attention")
            pretrained_dict_fix[key] = v
        elif "self_attention.in_proj_bias" in k:
            key = k.replace("self_attention.in_proj_bias", "attention.qkv.bias")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            key = key.replace("self_attention", "attention")
            pretrained_dict_fix[key] = v
        elif "self_attention.out_proj.weight" in k:
            key = k.replace("self_attention.out_proj.weight", "attention.out.weight")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            key = key.replace("self_attention", "attention")
            pretrained_dict_fix[key] = v
        elif "self_attention.out_proj.bias" in k:
            key = k.replace("self_attention.out_proj.bias", "attention.out.bias")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            key = key.replace("self_attention", "attention")
            pretrained_dict_fix[key] = v
        elif "mlp.linear_1" in k:
            key = k.replace("mlp.linear_1", "mlp.0")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            pretrained_dict_fix[key] = v
        elif "mlp.linear_2" in k:
            key = k.replace("mlp.linear_2", "mlp.3")
            key = key.replace("encoder.layers.encoder_layer_", "encoder.")
            pretrained_dict_fix[key] = v
        elif "encoder.ln.weight" in k:
            key = k.replace("encoder.ln.weight", "ln.weight")
            pretrained_dict_fix[key] = v
        elif "encoder.ln.bias" in k:
            key = k.replace("encoder.ln.bias", "ln.bias")
            pretrained_dict_fix[key] = v
        elif "heads.head.weight" in k:
            key = k.replace("heads.head.weight", "head.weight")
            pretrained_dict_fix[key] = v
        elif "heads.head.bias" in k:
            key = k.replace("heads.head.bias", "head.bias")
            pretrained_dict_fix[key] = v
        else:
            print(k)
            print("Not found")
            print()

    for k, v in model_dict.items():
        if k not in pretrained_dict_fix:
            print(k)

    # Update the current model's weights with the pretrained weights
    model_dict.update(pretrained_dict_fix)
    # Load the new weights into the model
    model.load_state_dict(model_dict)

    model.head = nn.Linear(model.head.in_features, 12)

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002)

    saved_acc = 90
    train_acc = 90
    saved = False

    # Train the model
    # num_epochs = 50
    num_epochs = 80
    schduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_loss += loss.item()

        test_accuracy = 100 * correct / total
        test_loss /= len(test_loader)

        if test_accuracy >= saved_acc:
            if test_accuracy == saved_acc:
                if train_accuracy > train_acc:
                    torch.save(model.state_dict(), 'cat_trans.pth')
                    saved_acc = test_accuracy
                    train_acc = train_accuracy
                    saved = True
            else:
                torch.save(model.state_dict(), 'cat_trans.pth')
                saved_acc = test_accuracy
                saved = True

        schduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save the model
    # torch.save(model.state_dict(), 'cat.pth')
    if not saved:
        torch.save(model.state_dict(), 'cat_trans.pth')

model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=1000)
model_dict = model.state_dict()
model.head = nn.Linear(model.head.in_features, 12)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Evaluate the model on the test set
# load cat.pth
model.load_state_dict(torch.load('cat_trans.pth'))

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        test_loss += loss.item()

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)

print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")