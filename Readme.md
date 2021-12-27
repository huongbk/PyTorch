  ### Table of Contents

Reference: <https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e>

---

- [x] A Simple Regression Problem
- [x] Gradient Descent
- [x] Linear Regression in Numpy
- [x] PyTorch
- [x] Autograd
- [x] Dynamic Computation Graph
- [x] Optimizer
- [x] Loss
- [x] Model
- [x] Dataset
- [x] DataLoader
- [x] Evaluation

---

**A Simple Regression Problem**

- Data Generation

  - Let’s start generating some synthetic data: we start with a vector of 100 points for our feature x and create our labels using a = 1, b = 2 and some Gaussian noise.
  - Next, let’s split our synthetic data into train and validation sets, shuffling the array of indices and using the first 80 shuffled points for training.

    ```python
    import numpy as np

    # Data Generation
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    # Uses first 80 random indices for train
    train_idx = idx[:80]
    # Uses the remaining indices for validation
    val_idx = idx[80:]

    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    ```

**Gradient Descent**

- [See here...](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#:~:text=the%20training%20set%E2%80%A6-,Gradient%20Descent,-If%20you%20are)

- Step 1: Compute the Loss

  - For a regression problem, the loss is given by the Mean Square Error (MSE), that is, the average of all squared differences between labels (y) and predictions (a + bx).

- Step 2: Compute the Gradients

  - A derivative tells you how much a given quantity changes when you slightly vary some other quantity. In our case, how much does our MSE loss change when we vary each one of our two parameters?

- Step 3: Update the Parameters

  - In the final step, we use the gradients to update the parameters. Since we are trying to minimize our losses, we reverse the sign of the gradient for the update.
  - There is still another parameter to consider: the learning rate, denoted by the Greek letter eta (that looks like the letter n), which is the multiplicative factor that we need to apply to the gradient for the parameter update.

- Step 4: Rinse and Repeat!

  - Now we use the updated parameters to go back to Step 1 and restart the process.

    `An epoch is complete whenever every point has been already used for computing the loss. For batch gradient descent, this is trivial, as it uses all points for computing the loss — one epoch is the same as one update. For stochastic gradient descent, one epoch means N updates, while for mini-batch (of size n), one epoch has N/n updates.`

  - _Repeating this process over and over, for many epochs, is, in a nutshell, training a model._

**Linear Regression in Numpy**

- It’s time to implement our linear regression model using gradient descent using Numpy only.

- For training a model, there are two initialization steps:

  - Random initialization of parameters/weights (we have only two, a and b)
  - Initialization of hyper-parameters (in our case, only learning rate and number of epochs)

- For each epoch:

  - Compute model’s predictions
  - Compute the loss, using predictions and and labels and the appropriate loss function for the task at hand
  - Compute the Gradients for every parameter
  - Update parameter

```python
# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Computes our model's predicted output
    yhat = a + b * x_train

    # How wrong is our model? That's the error!
    error = (y_train - yhat)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])
```

**PyTorch**

- [Tensor](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_tensor.html)

  - In Numpy, you may have an array that has three dimensions, right? That is, technically speaking, a tensor.
  - A scalar (a single number) has zero dimensions, a vector has one dimension, a matrix has two dimensions and a tensor has three or more dimensions. That’s it!
  - The biggest difference between a numpy array and a PyTorch Tensor is that a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU, just cast the Tensor to a cuda datatype.
  - `In PyTorch, every method that ends with an underscore (_) makes changes in-place, meaning, they will modify the underlying variable.`

- Loading Data, Devices and CUDA

  ```python
  import torch
  import torch.optim as optim
  import torch.nn as nn
  from torchviz import make_dot

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
  # and then we send them to the chosen device
  x_train_tensor = torch.from_numpy(x_train).float().to(device)
  y_train_tensor = torch.from_numpy(y_train).float().to(device)

  # Here we can see the difference - notice that .type() is more useful
  # since it also tells us WHERE the tensor is (device)
  print(type(x_train), type(x_train_tensor), x_train_tensor.type())
  ```

- Creating Parameters

  - The latter tensors require the computation of its gradients, so we can update their values (the parameters’ values, that is). That’s what the `requires_grad=True` argument is good for. It tells PyTorch we want it to compute gradients for us.

    ```python
    # FIRST
    # Initializes parameters "a" and "b" randomly, ALMOST as we did in Numpy
    # since we want to apply gradient descent on these parameters, we need
    # to set REQUIRES_GRAD = TRUE
    a = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    print(a, b)

    # SECOND
    # But what if we want to run it on a GPU? We could just send them to device, right?
    a = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    print(a, b)
    # Sorry, but NO! The to(device) "shadows" the gradient...

    # THIRD
    # We can either create regular tensors and send them to the device (as we did with our data)
    a = torch.randn(1, dtype=torch.float).to(device)
    b = torch.randn(1, dtype=torch.float).to(device)
    # and THEN set them as requiring gradients...
    a.requires_grad_()
    b.requires_grad_()
    print(a, b)
    ```

- or

  ```python
  # We can specify the device at the moment of creation - RECOMMENDED!
  torch.manual_seed(42)
  a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  print(a, b)
  ```

**Autograd**

- [See more...](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#)
- Autograd is PyTorch’s automatic differentiation package. Thanks to it, we don’t need to worry about partial derivatives, chain rule or anything like it.
- So, how do we tell PyTorch to do its thing and compute all gradients? That’s what `backward()` is good for.

```python
lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # No more manual computation of gradients!
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()

    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()
    # Let's check the computed gradients...
    print(a.grad)
    print(b.grad)

    # What about UPDATING the parameters? Not so fast...

    # FIRST ATTEMPT
    # AttributeError: 'NoneType' object has no attribute 'zero_'
    # a = a - lr * a.grad
    # b = b - lr * b.grad
    # print(a)

    # SECOND ATTEMPT
    # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    # a -= lr * a.grad
    # b -= lr * b.grad

    # THIRD ATTEMPT
    # We need to use NO_GRAD to keep the update out of the gradient computation
    # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()

print(a, b)
```

- In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels. Let’s walk through a small example to demonstrate this. As before, we load a pretrained resnet18 model, and freeze all the parameters.

  ```python
  from torch import nn, optim

  model = torchvision.models.resnet18(pretrained=True)

  # Freeze all the parameters in the network
  for param in model.parameters():
      param.requires_grad = False
  ```

**Dynamic Computation Graph**

- [See here...](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#3806)
- The [PyTorchViz](https://github.com/szagoruyko/pytorchviz) package and its `make_dot(variable)` method allows us to easily visualize a graph associated with a given Python variable.

**Optimizer**

- An optimizer takes the parameters we want to update, the learning rate we want to use (and possibly many other hyper-parameters as well!) and performs the updates through its `step()` method.
- Besides, we also don’t need to zero the gradients one by one anymore. We just invoke the optimizer’s `zero_grad()` method and that’s it!

- In the code below, we create a `Stochastic Gradient Descent (SGD)` optimizer to update our parameters a and b.

  ```python
  torch.manual_seed(42)
  a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  print(a, b)

  lr = 1e-1
  n_epochs = 1000

  # Defines a SGD optimizer to update the parameters
  optimizer = optim.SGD([a, b], lr=lr)

  for epoch in range(n_epochs):
      yhat = a + b * x_train_tensor
      error = y_train_tensor - yhat
      loss = (error ** 2).mean()

      loss.backward()

      # No more manual update!
      # with torch.no_grad():
      #     a -= lr * a.grad
      #     b -= lr * b.grad
      optimizer.step()

      # No more telling PyTorch to let gradients go!
      # a.grad.zero_()
      # b.grad.zero_()
      optimizer.zero_grad()

  print(a, b)
  ```

  ```python
  # BEFORE: a, b
  tensor([0.6226], device='cuda:0', requires_grad=True)
  tensor([1.4505], device='cuda:0', requires_grad=True)
  # AFTER: a, b
  tensor([1.0235], device='cuda:0', requires_grad=True)
  tensor([1.9690], device='cuda:0', requires_grad=True)
  ```

**Loss**

- We now tackle the loss computation. As expected, PyTorch got us covered once again. There are many loss functions to choose from, depending on the task at hand.
- Since ours is a regression, we are using the `Mean Square Error (MSE)` loss.

  ```text
  Notice that nn.MSELoss actually creates a loss function for us — it is NOT the loss function itself.
  Moreover, you can specify a reduction method to be applied
  That is, how do you want to aggregate the results for individual points
  You can average them (reduction=’mean’) or simply sum them up (reduction=’sum’).
  ```

  ```python
  torch.manual_seed(42)
  a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
  print(a, b)

  lr = 1e-1
  n_epochs = 1000

  # Defines a MSE loss function
  loss_fn = nn.MSELoss(reduction='mean')

  optimizer = optim.SGD([a, b], lr=lr)

  for epoch in range(n_epochs):
      yhat = a + b * x_train_tensor

      # No more manual loss!
      # error = y_tensor - yhat
      # loss = (error ** 2).mean()
      loss = loss_fn(y_train_tensor, yhat)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

  print(a, b)
  ```

**Model**

- In PyTorch, a model is represented by a regular Python class that inherits from the Module class.
- Moreover, we can get the current values for all parameters using our model’s `state_dict()` method.
- `IMPORTANT:` we need to send our model to the same device where the data is. If our data is made of GPU tensors, _our model must “live” inside the GPU as well._

  ```python
  torch.manual_seed(42)

  # Now we can create a model and send it at once to the device
  model = ManualLinearRegression().to(device)
  # We can also inspect its parameters using its state_dict
  print(model.state_dict())

  lr = 1e-1
  n_epochs = 1000

  loss_fn = nn.MSELoss(reduction='mean')
  optimizer = optim.SGD(model.parameters(), lr=lr)

  for epoch in range(n_epochs):
      # What is this?!?
      model.train()

      # No more manual prediction!
      # yhat = a + b * x_tensor
      yhat = model(x_train_tensor)

      loss = loss_fn(y_train_tensor, yhat)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

  print(model.state_dict())
  ```

- In PyTorch, models have a `train()` method which, somewhat disappointingly, does NOT perform a training step. Its only purpose is to set the model to training mode. Why is this important? Some models may use mechanisms like `Dropout`, for instance, which have distinct behaviors in training and evaluation phases.

- `Nested Models`

  - Let’s use PyTorch’s Linear model as an attribute of our own, thus creating a nested model.

    - In the `__init__` method, we created an attribute that contains our nested Linear model.
    - In the `forward()` method, we call the nested model itself to perform the forward pass _(notice, we are **not** calling self.linear.forward(x)!)._

      ```python
      class LayerLinearRegression(nn.Module):
          def __init__(self):
              super().__init__()
              # Instead of our custom parameters
              # we use a Linear layer with single input and single output
              self.linear = nn.Linear(1, 1)

          def forward(self, x):
              # Now it only takes a call to the layer to make predictions
              return self.linear(x)
      ```

    - Now, if we call the `parameters()` method of this model, PyTorch will figure the parameters of its attributes in a recursive way.
    - You can try it yourself using something like: `[*LayerLinearRegression().parameters()]` to get a list of all parameters.
    - You can also add new Linear attributes and, even if you don’t use them at all in the forward pass, they will still be listed under `parameters()`.

- `Sequential Models`

  - For straightforward models, that use run-of-the-mill layers, where the output of a layer is sequentially fed as an input to the next, we can use a, er… Sequential model :-)
  - In our case, we would build a Sequential model with a single argument, that is, the `Linear` layer we used to train our linear regression. The model would look like this:

    ```python
    # Alternatively, you can use a Sequential model
    model = nn.Sequential(nn.Linear(1, 1)).to(device)
    ```

- Training Step

  - So far, we’ve defined an `optimizer`, a `loss function` and a `model`. Scroll up a bit and take a quick look at the code inside the loop. Would it change if we were using a different `optimizer`, or `loss`, or even `model`? If not, how can we make it more generic?

    - Well, I guess we could say all these lines of code perform a training step, given those three elements (`optimizer`, `loss` and `model`),the `features` and the `labels`.
    - So, how about writing a function that takes those `three elements` and `returns another function` that performs a training step, taking a set of features and labels as arguments and returning the corresponding loss?
    - Then we can use this general-purpose function to build a `train_step()` function to be called inside our training loop. Now our code should look like this… see how tiny the training loop is now?

      ```python
      def make_train_step(model, loss_fn, optimizer):
          # Builds function that performs a step in the train loop
          def train_step(x, y):
              # Sets model to TRAIN mode
              model.train()
              # Makes predictions
              yhat = model(x)
              # Computes loss
              loss = loss_fn(y, yhat)
              # Computes gradients
              loss.backward()
              # Updates parameters and zeroes gradients
              optimizer.step()
              optimizer.zero_grad()
              # Returns the loss
              return loss.item()

          # Returns the function that will be called inside the train loop
          return train_step

      # Creates the train_step function for our model, loss function and optimizer
      train_step = make_train_step(model, loss_fn, optimizer)
      losses = []

      # For each epoch...
      for epoch in range(n_epochs):
          # Performs one train step and returns the corresponding loss
          loss = train_step(x_train_tensor, y_train_tensor)
          losses.append(loss)

      # Checks model's parameters
      print(model.state_dict())
      ```

**Dataset**

- In PyTorch, a `dataset` is represented by a regular `Python class` that inherits from the `Dataset` class. You can think of it as a kind of a Python `list of tuples`, each tuple corresponding to one point ` (features, label)`.
- [_Creating a Custom Dataset for your files_](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

  - A custom Dataset class must implement three functions: `__init__`, `__len__`, and `__getitem__`

  - Let’s build a simple custom dataset that takes `two tensors as arguments`: `one` for the features, `one` for the labels.

  - For any given index, our dataset class will return the corresponding slice of each of those tensors. It should look like this:

    ```python
    from torch.utils.data import Dataset, TensorDataset

    class CustomDataset(Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor

        def __getitem__(self, index):
            return (self.x[index], self.y[index])

        def __len__(self):
            return len(self.x)

    # Wait, is this a CPU tensor now? Why? Where is .to(device)?
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])
    ```

**DataLoader**

- So we use PyTorch’s DataLoader class for this job. We tell it which dataset to use (the one we just built in the previous section), the desired mini-batch size and if we’d like to shuffle it or not. That’s it!

  ```python
  from torch.utils.data import DataLoader

  train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
  ```

- ex

  ```python
  # importing the required libraries
  import torch
  from torch.utils.data import Dataset
  from torch.utils.data import DataLoader

  # defining the Dataset class
  class data_set(Dataset):
      def __init__(self):
          numbers = list(range(0, 100, 1))
          self.data = numbers

      def __len__(self):
          return len(self.data)

      def __getitem__(self, index):
          return self.data[index]


  dataset = data_set()

  # implementing dataloader on the dataset and printing per batch
  dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
  for i, batch in enumerate(dataloader):
      print(i, batch)
  ```

- `Random Split`

  - PyTorch’s `random_split()` method is an easy and familiar way of performing a training-validation split. Just keep in mind that, in our example, we need to apply it to the whole dataset (not the training dataset we built in two sections ago).

    ```python
    from torch.utils.data.dataset import random_split

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [80, 20])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)
    ```

**Evaluation**

- This is the last part of our journey — we need to change the training loop to include the evaluation of our model, that is, computing the validation loss.
- There are `two small, yet important`, things to consider:

  - `torch.no_grad()`: even though it won’t make a difference in our simple model, it is a good practice to wrap the validation inner loop with this context manager to disable any gradient calculation that you may inadvertently trigger — _gradients belong in training, not in validation steps;_
  - `eval()`: the only thing it does is _setting the model to evaluation mode_ (just like its `train()` counterpart did), so the model can adjust its behavior regarding some operations, like `Dropout`.

    ```python
    losses = []
    val_losses = []
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                model.eval()

                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

    print(model.state_dict())
    ```

- Full code

```python

torch.manual_seed(42)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Builds dataset with ALL data
dataset = TensorDataset(x_tensor, y_tensor)
# Splits randomly into train and validation datasets
train_dataset, val_dataset = random_split(dataset, [80, 20])
# Builds a loader for each dataset to perform mini-batch gradient descent
train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

# Builds a simple sequential model
model = nn.Sequential(nn.Linear(1, 1)).to(device)
print(model.state_dict())

# Sets hyper-parameters
lr = 1e-1
n_epochs = 150

# Defines loss function and optimizer
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []
val_losses = []
# Creates function to perform train step from model, loss and optimizer
train_step = make_train_step(model, loss_fn, optimizer)

# Training loop
for epoch in range(n_epochs):
    # Uses loader to fetch one mini-batch for training
    for x_batch, y_batch in train_loader:
        # NOW, sends the mini-batch data to the device
        # so it matches location of the MODEL
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # One stpe of training
        loss = train_step(x_batch, y_batch)
        losses.append(loss)

    # After finishing training steps for all mini-batches,
    # it is time for evaluation!

    # We tell PyTorch to NOT use autograd...
    # Do you remember why?
    with torch.no_grad():
        # Uses loader to fetch one mini-batch for validation
        for x_val, y_val in val_loader:
            # Again, sends data to same device as model
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # What is that?!
            model.eval()
            # Makes predictions
            yhat = model(x_val)
            # Computes validation loss
            val_loss = loss_fn(y_val, yhat)
            val_losses.append(val_loss.item())

print(model.state_dict())
print(np.mean(losses))
print(np.mean(val_losses))
```
