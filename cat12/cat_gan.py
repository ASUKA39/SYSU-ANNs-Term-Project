import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import random
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
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Create custom datasets for training and testing
train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)

# Create data loaders for training and testing
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Bottleneck block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out

# Define the ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
if not test_flag:
    # Download the pretrained weights
    # !wget https://download.pytorch.org/models/resnet50-19c8e357.pth
    if not os.path.exists('resnet50-19c8e357.pth'):
        os.system('wget https://download.pytorch.org/models/resnet50-19c8e357.pth')

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 定义生成器和鉴别器，并将它们移动到 GPU 上
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3, img_height=64, img_width=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.img_height = img_height
        self.img_width = img_width

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 输出通道数修改为 num_channels
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.model = ResNet50()
        model_dict = self.model.state_dict()
        if not test_flag:
            pretrained_dict = torch.load('resnet50-19c8e357.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

# 实例化生成器和鉴别器，并将它们移动到 GPU 上
generator = Generator(latent_dim=100).to(device)
discriminator = Discriminator(num_classes=1000).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
# optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizer_d = optim.SGD(discriminator.parameters(), lr=0.001, momentum=0.9)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

latent_dim = 100
num_epochs = 20
counter = 0

if not test_flag:

    # Train the model
    for epoch in range(num_epochs):
        total_real_loss = 0.0
        total_fake_loss = 0.0
        correct_real = 0
        correct_fake = 0

        for real_images, labels in train_loader:
            counter += 1
            # print(counter)
            real_images, labels = real_images.cuda(), labels.cuda()

            # 训练鉴别器
            optimizer_d.zero_grad()
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, labels)
            real_loss.backward()

            # 生成一些假的图像
            fake_noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(fake_noise)

            # 假图像的标签（你可以根据需要定义）
            # fake_labels = torch.randint(0, 12, (batch_size,)).to(device)
            fake_labels = torch.randint(13, 1000, (batch_size,)).to(device)

            # 鉴别器对真图像的输出
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, labels)
            total_real_loss += real_loss.item()

            # 鉴别器对假图像的输出
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            total_fake_loss += fake_loss.item()

            # 计算准确率
            _, predicted_real = torch.max(real_outputs.data, 1)
            correct_real += (predicted_real == labels).sum().item()

            _, predicted_fake = torch.max(fake_outputs.data, 1)
            # correct_fake += (predicted_fake == fake_labels).sum().item()
            for i in range(len(predicted_fake)):
                if predicted_fake[i] not in range(1, 13):
                    correct_fake += 1

            optimizer_d.step()

        # 输出每个 epoch 的信息
        avg_real_loss = total_real_loss / len(train_loader)
        avg_fake_loss = total_fake_loss / len(train_loader)
        accuracy_real = correct_real / len(train_loader.dataset)
        accuracy_fake = correct_fake / len(train_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}: Real Loss: {avg_real_loss:.4f}, Fake Loss: {avg_fake_loss:.4f}, ', end='')
        print(f'Real Acc: {accuracy_real*100:.2f}%, Fake Acc: {accuracy_fake*100:.2f}%')

        # 保存模型参数
        checkpoint_path = f'./cat_GAN.pth'
        torch.save(discriminator.state_dict(), checkpoint_path)

        # print(f'Model parameters saved to {checkpoint_path}')

# 在验证集上评估模型
discriminator.load_state_dict(torch.load('cat_GAN.pth'))
discriminator.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = discriminator(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
