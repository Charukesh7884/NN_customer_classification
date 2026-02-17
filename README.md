# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="901" height="905" alt="image" src="https://github.com/user-attachments/assets/ef64d497-8caf-4d35-b74a-25043fdcee46" />

## DESIGN STEPS

## STEP 1:
Import necessary libraries and load the dataset.

## STEP 2:
Encode categorical variables and normalize numerical features.

## STEP 3:
Split the dataset into training and testing subsets.

## STEP 4:
Design a multi-layer neural network with appropriate activation functions.

## STEP 5:
Train the model using an optimizer and loss function.

## STEP 6:
Evaluate the model and generate a confusion matrix.

## STEP 7:
Use the trained model to classify new data samples.

## STEP 8:
Display the confusion matrix, classification report, and predictions.
## PROGRAM

### Name: CHARUKESH S
### Register Number: 212224230044

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)  # 4 output classes (A, B, C, D)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc3(x)  # No activation, CrossEntropyLoss applies Softmax internally
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_model(model, train_loader, criterion, optimizer, epochs=100)
```
```python
#function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


```
## Dataset Information
<img width="997" height="737" alt="image" src="https://github.com/user-attachments/assets/90ce77f3-6e30-49af-8faf-8b27e804a56f" />

## OUTPUT
<img width="1125" height="211" alt="image" src="https://github.com/user-attachments/assets/758dc93a-3888-4256-960e-adff90b4681c" />

<img width="960" height="84" alt="image" src="https://github.com/user-attachments/assets/5ef763a1-17ae-45bc-8469-da2fd6b99852" />

<img width="1047" height="191" alt="image" src="https://github.com/user-attachments/assets/ce8a41ea-a789-40ab-a9a5-a2fbb0e5ca8b" />

### Confusion Matrix

<img width="729" height="610" alt="image" src="https://github.com/user-attachments/assets/a19e6ffe-e637-4e22-919b-2b85736d4e3d" />

### Classification Report
<img width="685" height="278" alt="image" src="https://github.com/user-attachments/assets/5c552cf5-bba8-46e2-afc3-1d7a03d1ba68" />

### New Sample Data Prediction
<img width="1086" height="105" alt="image" src="https://github.com/user-attachments/assets/73ab0f35-fa3a-4904-953f-94ddb4c0da75" />

## RESULT
Thus neural network classification model is developded for the given dataset.
