import numpy as np

from CNN_Functions import *
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
import os


Set_Seed(seed=0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X = np.load('...MNIST_data_file...')
Y = np.load('...MNIST_label_file...').astype(int)
X = Norm_Data(X)
Y = np.squeeze(Y)

X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

Classes = len(np.unique(Y))

CNN = CNN_Model(K=Classes, input_dims=X_train.shape[1:], filter1=32, filter2=64, filter3=128, filter4=256, lin1=256)

model_location = os.path.join('CNN_Model_Save', 'CNN_model_classes-10.pt')
CNN.load_state_dict(torch.load(model_location))

preds = Infer(X_test, CNN)
truth = Y_test
confusion = sm.confusion_matrix(truth, preds)
df_cm = pd.DataFrame(confusion)
df_cm.to_csv('Confusion_Matrix.csv', index=False)

plt.figure(figsize=(8, 8))
sn.heatmap(df_cm, annot=True, fmt='.0f')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.savefig('Confusion_Matrix.png')
plt.show()





