import numpy as np
import random
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
import sklearn.metrics as sm
import seaborn as sn
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc
import sys


class CNN_Model(nn.Module):
    def __init__(self, K, input_dims, filter1, filter2, filter3, filter4, lin1):
        super(CNN_Model, self).__init__()

        self.input_dims = input_dims

        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.filter4 = filter4
        # self.filter5 = filter5

        self.lin1 = lin1

        self.conv1 = nn.Conv2d(input_dims[0], self.filter1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filter1, self.filter2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.filter2, self.filter3, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.filter3, self.filter4, 3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(self.filter4, self.filter5, 3, stride=1, padding=1)

        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool3 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        # self.maxpool5 = torch.nn.MaxPool2d(3, stride=2, padding=1)

        neurons, self.shape_pre_flatten = self.getNeuronNum(torch.zeros(input_dims).unsqueeze(0))

        self.linear1 = nn.Linear(neurons, lin1)
        self.linear = nn.Linear(lin1, K)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)

        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)

        # x = F.relu(self.conv5(x))
        # x = self.maxpool5(x)

        flat_x = torch.flatten(x, 1)
        flat_x = F.relu(self.linear1(flat_x))
        pred = F.softmax(self.linear(flat_x), dim=1)

        return pred

    def getNeuronNum(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)

        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)

        # x = F.relu(self.conv5(x))
        # x = self.maxpool5(x)

        flat_x = torch.flatten(x, 1)

        return flat_x.numel(), x.shape

    def Shape(self, x):

        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.maxpool1(x)
        print(x.shape)

        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.maxpool2(x)
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)
        x = self.maxpool3(x)
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)
        x = self.maxpool4(x)
        print(x.shape)

        # x = F.relu(self.conv5(x))
        # print(x.shape)
        # x = self.maxpool5(x)
        # print(x.shape)

        flat_x = torch.flatten(x, 1)
        print(flat_x.shape)
        flat_x = F.relu(self.linear1(flat_x))
        print(flat_x.shape)
        pred = F.softmax(self.linear(flat_x), dim=1)
        print(pred.shape)

        sys.exit()


def train_CNN(_network, _X_train, _X_val, _Y_train, _Y_val, epochs=10, batch_size=64, lr=1e-4, device='GPU'):

    _X_train = torch.tensor(_X_train, dtype=torch.float32)
    _X_val = torch.tensor(_X_val, dtype=torch.float32)

    _Y_train = torch.tensor(_Y_train, dtype=torch.long)
    _Y_val = torch.tensor(_Y_val, dtype=torch.long)

    if device == 'GPU':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = torch.optim.Adam(_network.parameters(), lr=lr, weight_decay=0)
    ce_loss = nn.CrossEntropyLoss()

    _network.to(device)
    _network.train()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    training_set = TensorDataset(_X_train, _Y_train)
    val_set = TensorDataset(_X_val, _Y_val)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    train_losses = []
    validation_losses = []
    for _ in tqdm(range(epochs)):

        for batch in train_loader:

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            preds = _network.forward(images)
            train_loss = ce_loss(preds, labels)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            train_losses.append(train_loss.item())

        _network.eval()
        for batch in validation_loader:

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            preds = _network.forward(images)
            val_loss = ce_loss(preds, labels)
            validation_losses.append(val_loss.item())

    return train_losses, validation_losses


def Norm_Data(_X):

    _W = np.zeros(_X.shape, dtype=float)

    for c in range(_X.shape[1]):
        for i in range(len(_W)):
            _W[i, c] = (_X[i, c] - _X[i, c].min()) / (_X[i, c].max() - _X[i, c].min())

    return _W


def Create_RGB_Image_With_Masks(_X):

    _W = np.zeros(_X.shape)

    for i in range(len(_X)):

        for j in range(3):
            _X[j, 0] = (_X[j, 0] - _X[j, 0].min())/(_X[j, 0].max() - _X[j, 0].min())

        _W[i, 0] = _X[i, 0]
        _W[i, 1] = _X[i, 0] * _X[i, 1]
        _W[i, 2] = _X[i, 0] * ((_X[i, 2] - _X[i, 1]) > 0)

    return _W


def Set_Seed(seed=None, seed_torch=True, verbose=False):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if verbose:
        print(f'Random seed {seed} has been set.')


def Set_Device(verbose=False):

    n_gpu = torch.cuda.device_count()

    if verbose:
        print('GPUs found:', n_gpu)

    for i in range(n_gpu):
        device_name = 'cuda:' + str(i)
        device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        print(device, torch.cuda.get_device_name(i))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device


def Split_Train_Val_Test(_X, _Y, frac_train=0.64, frac_val=0.8):

    idx = 0

    _, h, w = _X.shape

    _X_train = np.zeros([0, h, w])
    _X_val = np.zeros([0, h, w])
    _X_test = np.zeros([0, h, w])

    _Y_train = np.zeros([0])
    _Y_val = np.zeros([0])
    _Y_test = np.zeros([0])

    for i in range(int(_Y.max() + 1)):

        N = sum(_Y == i)
        idx_new = N + idx

        X_sec = _X[idx:idx_new]
        Y_sec = _Y[idx:idx_new]

        mark_train = int(frac_train * len(X_sec))
        mark_val = int(frac_val * len(X_sec))

        _X_train = np.concatenate((_X_train, X_sec[:mark_train]))
        _X_val = np.concatenate((_X_val, X_sec[mark_train:mark_val]))
        _X_test = np.concatenate((_X_test, X_sec[mark_val:]))

        _Y_train = np.concatenate((_Y_train, Y_sec[:mark_train]))
        _Y_val = np.concatenate((_Y_val, Y_sec[mark_train:mark_val]))
        _Y_test = np.concatenate((_Y_test, Y_sec[mark_val:]))

        idx = idx_new

    return _X_train, _X_val, _X_test, _Y_train, _Y_val, _Y_test


def Split_Train_Val_Test_RGB(_X, _Y, frac_train=0.64, frac_val=0.8):

    idx = 0
    _, c, h, w = _X.shape

    _X_train = np.zeros([0, c, h, w])
    _X_val = np.zeros([0, c, h, w])
    _X_test = np.zeros([0, c, h, w])

    _Y_train = np.zeros([0])
    _Y_val = np.zeros([0])
    _Y_test = np.zeros([0])

    for i in range(int(_Y.max() + 1)):

        N = sum(_Y == i)
        idx_new = N + idx

        X_sec = _X[idx:idx_new]
        Y_sec = _Y[idx:idx_new]

        mark_train = int(frac_train * len(X_sec))
        mark_val = int(frac_val * len(X_sec))

        _X_train = np.concatenate((_X_train, X_sec[:mark_train]))
        _X_val = np.concatenate((_X_val, X_sec[mark_train:mark_val]))
        _X_test = np.concatenate((_X_test, X_sec[mark_val:]))

        _Y_train = np.concatenate((_Y_train, Y_sec[:mark_train]))
        _Y_val = np.concatenate((_Y_val, Y_sec[mark_train:mark_val]))
        _Y_test = np.concatenate((_Y_test, Y_sec[mark_val:]))

        idx = idx_new

    return _X_train, _X_val, _X_test, _Y_train, _Y_val, _Y_test


def Shuffle(_X, _Y):

    idx = np.arange(len(_X))

    random.shuffle(idx)

    _X = _X[idx]
    _Y = _Y[idx]

    return _X, _Y


def Infer(_images, _network):

    _network.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _network.to(device)

    predictions = np.empty([len(_images)])
    for i, im in tqdm(enumerate(_images)):

        image = torch.Tensor(np.expand_dims(im, axis=0)).to(device)
        predictions[i] = np.argmax(_network.forward(image).cpu().detach().numpy())

    return np.array(predictions, dtype=int)


def Confusion_Matrix(CNN, model_location, X_test, Y_test, column_names):
    # model_location = network_name + '.pt'
    CNN.load_state_dict(torch.load(model_location))

    preds = Infer(X_test, CNN)
    truth = Y_test.numpy()

    # preds = cf.infer(W, CNN)
    # truth = Y

    confusion = sm.confusion_matrix(truth, preds)

    accuracy = sm.accuracy_score(truth, preds)
    precision = sm.precision_score(truth, preds, average='macro')
    recall = sm.recall_score(truth, preds, average='macro')
    f1 = 2 * (precision * recall) / (precision + recall)

    print(confusion)
    print(accuracy, precision, recall, f1)

    # column_names = ['CEL', 'ER', 'MIT', 'NUC', 'NUCI', 'NUCM']
    # column_names = ['ER', 'MIT']

    df_cm = pd.DataFrame(confusion, index=column_names, columns=column_names)
    plt.figure(figsize=(6, 6))
    sn.heatmap(df_cm, annot=True, fmt='.0f')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()