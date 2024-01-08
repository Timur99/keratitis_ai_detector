from time import time
from tqdm import tqdm

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms

from torcheval.metrics.functional import multiclass_f1_score
torch.manual_seed(0)
print('CUDA', torch.cuda.is_available())
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformer
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create Dataset
TRAIN_ROOT = "eyedataset/train"
TEST_ROOT = "eyedataset/val"

train_ds = ImageFolder(TRAIN_ROOT, transform=tfm)
test_ds = ImageFolder(TEST_ROOT, transform=tfm)

# Length of Train and Test Datasets
LEN_TRAIN = len(train_ds)
LEN_TEST = len(test_ds)
print(LEN_TRAIN, LEN_TEST)

# Index Mapping
print(train_ds.class_to_idx)

# Data Loader
train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=20, shuffle=True)

# Model
model = resnet18(weights=True)

# Replace Output of Fully Connected Layer with Number of Labels for our Classification Problem
# aka 3 (normal, tired, keratit)
model.fc = Linear(in_features=512, out_features=3)
model = model.to(device)

# Optimiser
optimiser = Adam(model.parameters(), lr=3e-4, weight_decay=0.0001)

# Loss Function
loss_fn = CrossEntropyLoss()

for epoch in range(6):
    start = time()

    tr_acc = 0
    test_acc = 0

    # Train
    model.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        for xtrain, ytrain in tepoch:
            optimiser.zero_grad()

            xtrain = xtrain.to(device)
            train_prob = model(xtrain)
            train_prob = train_prob.cpu()

            loss = loss_fn(train_prob, ytrain)
            loss.backward()
            optimiser.step()

            # training ends

            train_pred = torch.max(train_prob, 1).indices
            tr_acc += int(torch.sum(train_pred == ytrain))

        ep_tr_acc = tr_acc / LEN_TRAIN

    # Evaluate
    model.eval()
    with torch.no_grad():
        for xtest, ytest in test_loader:
            xtest = xtest.to(device)
            test_prob = model(xtest)
            test_prob = test_prob.cpu()

            test_pred = torch.max(test_prob, 1).indices
            print(multiclass_f1_score(test_pred, ytest, num_classes=3))
            test_acc += int(torch.sum(test_pred == ytest))

        ep_test_acc = test_acc / LEN_TEST

    end = time()
    duration = (end - start) / 60

    print(
        f"Epoch: {epoch}, Time: {duration}, Loss: {loss}\nTrain_acc: {ep_tr_acc}, Test_acc: {ep_test_acc}")

torch.save(model, 'model.pkl')
