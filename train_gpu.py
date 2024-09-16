import cupy as np
import pandas as pd

np.random.seed(0)


##############################
#        Class: Layer        #
##############################
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regulizer_l1 = 0.0, bias_regulizer_l1 = 0.0, weight_regulizer_l2 = 0.0, bias_regulizer_l2 = 0.0):
        self.weights =  0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regulizer_l1 = weight_regulizer_l1
        self.bias_regulizer_l1 = bias_regulizer_l1
        self.weight_regulizer_l2 = weight_regulizer_l2
        self.bias_regulizer_l2 = bias_regulizer_l2
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regulizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regulizer_l1 * dl1

        if self.weight_regulizer_l2 > 0:
            self.dweights += 2 * self.weight_regulizer_l2 * self.weights
            
        if self.bias_regulizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regulizer_l1 * dl1

        if self.bias_regulizer_l2 > 0:
            self.dbiases += 2 * self.bias_regulizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


##############################
#        Class: ReLU         #
##############################
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



##############################
#        Class: Softmax      #
##############################
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)



##############################
#        Class: Loss         #
##############################
class Loss:
    def calculate(self, y_pred, y_true):
        sample_loss = self.forward(y_pred, y_true)
        average_loss = np.mean(sample_loss)
        return average_loss

    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regulizer_l1 > 0:
            regularization_loss += layer.weight_regulizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regulizer_l2 > 0:
            regularization_loss += layer.weight_regulizer_l2 * np.sum(np.square(layer.weights))
        if layer.bias_regulizer_l1 > 0:
            regularization_loss += layer.bias_regulizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regulizer_l2 > 0:
            regularization_loss += layer.bias_regulizer_l2 * np.sum(np.square(layer.biases))
        return regularization_loss
    


##############################
#  Class: Loss_CrossEntropy  #
##############################
class Loss_CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        if len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods
        


#################################################
#  Class: Activation_Softmax_Loss_CrossEntropy  #
#################################################
class Activation_Softmax_Loss_CrossEntropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        return self.loss.calculate(self.activation.output, y_true)
    
    # Need To Learn
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


##############################
#   Class: Optimizer_Adam    #
##############################
class Optimizer_Adam:
    def __init__(self, learning_rate = 1.0, decay = 0.0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
    
    def pre_update_params(self):
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Need to Learn
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1 ))
        bias_momentums_corrected = layer.bias_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        bias_cache_corrected = layer.bias_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1


def calculate_accuracy(y_pred, y_true):
    if len(y_true.shape)==2:
        y_true = np.argmax(y_true, axis=1)
    predicted_class = np.argmax(y_pred, axis=1)

    return np.mean(predicted_class==y_true)



##############################
#          Read Data         #
##############################
df = pd.read_csv('/raid/home/sahilm/NN_from_scratch/MNIST_from_Scratch/Data/mnist_train.csv')
y_train = df['label']
y_train = y_train.to_numpy()
df = df.drop(columns=['label'])
X_train = df.to_numpy()

df = pd.read_csv('/raid/home/sahilm/NN_from_scratch/MNIST_from_Scratch/Data/mnist_test.csv')
y_test = df['label']
y_test = y_test.to_numpy()
df = df.drop(columns=['label'])
X_test = df.to_numpy()

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


##############################
#   Normalization of Data    #
##############################
a = 0
b = 1
xminimum = np.min(X_train,axis=0)
xmaximum = np.max(X_train,axis=0)
range_of_x = np.double((xmaximum - xminimum).get())
range_of_x[range_of_x==0] = 1e-9
X_train = a + ( ((X_train - xminimum) * (b - a)) / np.asarray(range_of_x) )
X_test = a + ( ((X_test - xminimum) * (b - a)) / np.asarray(range_of_x) )

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std==0] = 1e-9
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


##############################
#        Define Model        #
##############################

layer1 = Layer_Dense(784, 100, weight_regulizer_l2=5e-4, bias_regulizer_l2=5e-4)
relu1 = Activation_ReLU()
layer2 = Layer_Dense(100, 10, weight_regulizer_l2=5e-4, bias_regulizer_l2=5e-4)
relu2 = Activation_ReLU()
layer3 = Layer_Dense(10, 10, weight_regulizer_l2=5e-4, bias_regulizer_l2=5e-4)

loss_activaion = Activation_Softmax_Loss_CrossEntropy()

optimizer_Adam = Optimizer_Adam(learning_rate = 0.005 , decay = 5e-7)

epochs = 2501

train_acc = []
train_loss = []
val_acc = []
val_loss = []
learning_rate = []

for epoch in range(epochs):

    # Calculate Validation Loss and Accuracy
    layer1.forward(X_test)
    relu1.forward(layer1.output)
    layer2.forward(relu1.output)
    relu2.forward(layer2.output)
    layer3.forward(relu2.output)

    y_pred_test = layer3.output
    data_loss = loss_activaion.forward(y_pred_test, y_test)
    regularization_loss = ( loss_activaion.loss.regularization_loss(layer1) + 
                            loss_activaion.loss.regularization_loss(layer2) + 
                            loss_activaion.loss.regularization_loss(layer3) 
                          )
    loss = data_loss + regularization_loss
    acc = calculate_accuracy(y_pred_test, y_test)
    val_loss.append(loss)
    val_acc.append(acc)


    # Forward Pass
    layer1.forward(X_train)
    relu1.forward(layer1.output)
    layer2.forward(relu1.output)
    relu2.forward(layer2.output)
    layer3.forward(relu2.output)

    y_pred_train = layer3.output
    data_loss = loss_activaion.forward(y_pred_train, y_train)
    regularization_loss = ( loss_activaion.loss.regularization_loss(layer1) + 
                            loss_activaion.loss.regularization_loss(layer2) + 
                            loss_activaion.loss.regularization_loss(layer3) 
                          )
    loss = data_loss + regularization_loss
    acc = calculate_accuracy(y_pred_train, y_train)
    train_loss.append(loss)
    train_acc.append(acc)
    learning_rate.append(optimizer_Adam.current_learning_rate)

    # Backward Pass
    loss_activaion.backward(y_pred_train, y_train)
    layer3.backward(loss_activaion.dinputs)
    relu2.backward(layer3.dinputs)
    layer2.backward(relu2.dinputs)
    relu1.backward(layer2.dinputs)
    layer1.backward(relu1.dinputs)

    # Update Parameters
    optimizer_Adam.pre_update_params()
    optimizer_Adam.update_params(layer1)
    optimizer_Adam.update_params(layer2)
    optimizer_Adam.update_params(layer3)
    optimizer_Adam.post_update_params()

    if epoch%100 == 0:
        print(f"Epoch:{epoch+1}   Train Acc: {train_acc[-1]:.4f}  Train Loss: {train_loss[-1]:.4f}    lr:{optimizer_Adam.current_learning_rate:.4f}")
        print(f"Epoch:{epoch+1}   Val Acc: {val_acc[-1]:.4f}  val Loss: {val_loss[-1]:.4f}")



train_acc = [np.asnumpy(array) for array in train_acc]
val_acc = [np.asnumpy(array) for array in val_acc]
train_loss = [np.asnumpy(array) for array in train_loss]
val_loss = [np.asnumpy(array) for array in val_loss]


# Plot Graphs

import matplotlib.pyplot as plt

# Assuming your accuracy and loss lists are already populated

# Create subplots for two plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))  # Adjust figsize as needed

# Plot accuracy on first subplot
ax1.plot(train_acc, label='Train Accuracy', color='blue')
ax1.plot(val_acc, label='Val Accuracy', color='indigo')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy')
ax1.legend()

# Plot loss on second subplot
ax2.plot(train_loss, label='Train Loss', color='red')
ax2.plot(val_loss, label='Val Loss', color='tomato')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss')
ax2.legend()

ax3.plot(learning_rate, label='learning_rate', color='green')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('learning_rate')
ax3.set_title('learning_rate')
ax3.legend()

# Make the overall plot layout tighter
plt.tight_layout()

# Don't display the plot (remove plt.show())
plt.savefig("MNIST_from_Scratch_GPU.png")
plt.close()
