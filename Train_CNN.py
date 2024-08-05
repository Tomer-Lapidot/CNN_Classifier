import matplotlib.pyplot as plt

from CNN_Functions import *
from sklearn.model_selection import train_test_split
import os

Set_Seed(seed=0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X = np.load('C:/Users/lapidott/PycharmProjects/pythonProject/CNN/MNIST/DigitMNIST/raw/data_digit_large.npy')
Y = np.load('C:/Users/lapidott/PycharmProjects/pythonProject/CNN/MNIST/DigitMNIST/raw/labels_digit.npy').astype(int)
X = Norm_Data(X)
Y = np.squeeze(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

Classes = len(np.unique(Y))

model_name = 'CNN_model_classes-' + str(Classes)
save_path = 'CNN_Model_Save'
if not (os.path.isdir(save_path)):
    os.mkdir(save_path)

CNN = CNN_Model(K=Classes, input_dims=X_train.shape[1:], filter1=32, filter2=64, filter3=128, filter4=256, lin1=256)

# Option to load checkpoint
# model_location = os.path.join(save_path, model_name + '.pt')
# CNN.load_state_dict(torch.load(model_location))

train_losses, validation_losses = train_CNN(CNN, X_train, X_val, Y_train, Y_val, epochs=50, batch_size=64, lr=5e-5)


plt.plot(np.arange(0, len(train_losses)), np.array(train_losses), c='blue', label='train')
plt.plot(4*np.arange(0, len(validation_losses)), np.array(validation_losses), c='orange', label='validation')
plt.legend()

torch.save(CNN.state_dict(), os.path.join(save_path, model_name + '.pt'))
np.save(model_name + '_training_losses.npy', train_losses)
np.save(model_name + '_validation_losses.npy', validation_losses)

# Plot losses
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='training')
plt.plot(np.arange(0, len(validation_losses) * 4, 4), validation_losses, label='validation')
plt.legend()
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('Loss_Curve.png')
plt.show()