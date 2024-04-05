import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = "1"

n_qubits = 4                # Number of qubits
step = 0.0004               # Learning rate
batch_size = 4              # Number of samples for each training step
num_epochs = 3              # Number of training epochs
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer

dev = qml.device("lightning.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Adjust data directory for AIDER dataset
data_dir = "AIDER/"  # Update this with the path to your AIDER dataset directory

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

class AIDER(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.as_tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, y_label


aider_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(240),
    transforms.ToTensor()])

squeeze_transforms = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.CenterCrop(140),
    transforms.ToTensor()])

# Load AIDER dataset
full_dataset = AIDER("multiclass_aider/aider_labels.csv", data_dir, transform=aider_transforms)

total_count = 6432
train_count = int(0.5 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_set, valid_set, test_set = torch.utils.data.random_split(full_dataset,
                                                                (train_count, valid_count, test_count))

# Initialize dataloader for AIDER dataset
dataloaders = {
    'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True),
    'val': torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
}

def H_layer(nqubits):

    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):

    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):

    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def quantum_net(q_input_features, q_weights_flat):

    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    H_layer(n_qubits)

    RY_layer(q_input_features)

    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)

# Define model architecture
class DressedQuantumNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 5)

    def forward(self, input_features):

        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            elem = elem.to('cpu')
            q_out_elem = quantum_net(elem, self.q_params)
            q_out_elem = torch.hstack(q_out_elem).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
            q_out = q_out.to(device)

        return self.post_net(q_out)

model_hybrid = torchvision.models.resnet18(pretrained=True)

model_hybrid.fc = DressedQuantumNet()

model_hybrid = model_hybrid.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=step)

exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler
)

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model_hybrid = train_model(model_hybrid, criterion, optimizer_hybrid, exp_lr_scheduler, dataloaders, device, num_epochs=10)