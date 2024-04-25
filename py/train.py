import torch
import jovian
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


dataframe_raw = pd.read_csv( "./car data.csv")
dataframe_raw.head()
# print(dataframe_raw)
your_name = "ChengxuWan" # at least 5 characters
def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.Kms_Driven=dataframe.Kms_Driven/10000
    # scale target
    # dataframe.Selling_Price = dataframe.Selling_Price *10.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['Car_Name'], axis=1)
    return dataframe

dataframe = customize_dataset(dataframe_raw, your_name)
del dataframe['Car_Name']
Age = 2023 - dataframe['Year']
dataframe.insert(1,'Age',Age)
del dataframe['Year']

# print(dataframe.head())
input_cols = ["Age","Present_Price","Kms_Driven","Seller_Type","Fuel_Type"]
categorical_cols = ["Fuel_Type","Seller_Type","Transmission"] 
output_cols = ["Selling_Price"]
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes


    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
print(dataframe)




inputs_array, targets_array = dataframe_to_arrays(dataframe)

inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [228, 57])
batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = len(input_cols)
output_size = len(output_cols)

class CarsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calcuate loss
        loss = F.l1_loss(out, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))

model = CarsModel()
list(model.parameters())
# model = CarsModel()

# print(list(model.parameters()))
# Eval algorithm
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Fitting algorithm
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    lowest_loss = float("inf")  # 初始化最低 loss 為正無窮大
    best_model = None  # 用於保存最低 loss 時的模型
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            current_loss = model.training_step(batch)
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if current_loss < lowest_loss:
                # 如果當前 loss 比最低 loss 還低，則更新最低 loss 和保存模型
                lowest_loss = current_loss
                best_model = model.state_dict()  # 保存模型參數
            # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    torch.save(best_model, "./best_model.pt")  # 保存整个模型
    return history

# Check the initial value that val_loss have
result = evaluate(model, val_loader)
print(result)
# Start with the Fitting
epochs = 3000
lr1 = 1e-3
history1 = fit(epochs, lr1, model, train_loader, val_loader)








