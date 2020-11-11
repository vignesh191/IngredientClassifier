import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IngredientGroupDataSet import IngredientGroupDataset

import pandas as pd

# cleaning the dataset
columns = ['INGREDIENT', 'FOOD_GROUP']
file = pd.read_csv('FoodGroup.csv', names=columns, header=0)
file = file[file['FOOD_GROUP'].notna()]

file.to_csv('./FoodGroup.csv')

test = file['FOOD_GROUP'].to_list()


#KEY FOR DATASET
# Vegetables - 0
# Fruits - 1
# Grains - 2
# Proteins - 3
# Other - 4

dataset = IngredientGroupDataset(csv_file='FoodGroup.csv', transform=None)
train_set, test_set = torch.utils.data.random_split(dataset, [5078, 1269])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=5078, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1269, shuffle=True)


#constructing the neural network
class Net(nn.Module):
   def __init__(self):
       super().__init__()
       self.fc1 = nn.Linear(30, 24)
       self.fc2 = nn.Linear(24, 24)
       self.fc3 = nn.Linear(24, 24)
       self.fc4 = nn.Linear(24, 5)

   def forward(self, data):
       data = F.relu(self.fc1(data))
       data = F.relu(self.fc2(data))
       data = F.relu(self.fc3(data))
       data = self.fc4(data)
       return F.log_softmax(data, dim=1)


net = Net()


optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3000

for epoch in range(EPOCHS):
    for data in train_loader: #data is a size 10 batch of labeled data
        X, y = data
        net.zero_grad() #begin with zero gradient
        output = net(X.float())
        loss = F.nll_loss(output, y) #loss func is this because y is one hot value
        loss.backward() #backpropagation
        optimizer.step()
    print(loss)



correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        X, y = data
        output = net(X.float())
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            total+=1

print("Accuracy: ", round(correct/total, 3))
