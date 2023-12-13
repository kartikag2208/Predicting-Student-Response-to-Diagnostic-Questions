from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data():
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {question_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {question_id: list,
        user_id: list, is_correct: list}
    """
    train_data = load_train_csv()
    train_matrix = load_train_sparse().toarray()
    valid_data = load_valid_csv()
    test_data = load_public_test_csv()

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data, train_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        # Apply sigmoid activation for g
        out = inputs
        g_output = F.sigmoid(self.g(inputs))
        
        # Apply sigmoid activation for h
        out = F.sigmoid(self.h(g_output))

        return out

def train_withoutl2(model, lr, train_matrix, train_data, zero_train_data, valid_data, num_epoch):

    model.train()
    training_acc = []
    validation_acc = []
    
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_acc = evaluate(model, zero_train_data, train_data)
        training_acc.append(train_acc)
        validation_acc.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    
    return training_acc, validation_acc

def train_withl2(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
   
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # loss = torch.sum((output - target) ** 2.)
            # loss.backward()

            # train_loss += loss.item()
            # optimizer.step()

            # Compute L2 regularization term
            regularization_term = lamb * model.get_weight_norm()

            # Compute the mean squared error loss
            loss = torch.sum((output - target) ** 2.) + regularization_term/2

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data, train_data = load_data()

    latent_dimensions = [10, 50, 100, 200, 500]
    # Set model hyperparameters.
    k = None
    model = None

    validation_acc = []
    # Set optimization hyperparameters.
    lr = 0.03
    num_epoch = 25
    lamb = None

    best_accuracy = 0.0
    best_k = None
    best_model = None

    for k in latent_dimensions:
        print("\nTraining with latent dimension k = {}".format(k))

        model = AutoEncoder(train_matrix.shape[1], k)
        train_withoutl2(model, lr, train_matrix, train_data, zero_train_matrix, valid_data, num_epoch)

        accuracy = evaluate(model, zero_train_matrix, valid_data)
        validation_acc.append(accuracy)
        print("Validation accuracy with k = {}: {}".format(k, accuracy))

        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_k = k
            best_model = model
    
    print("Best accuracy: {}, Best k = {}".format(best_accuracy, best_k))

    plt.scatter(latent_dimensions, validation_acc)
    plt.title("Validation accuracy v/s k")
    plt.xlabel("k(Latent dimension)")
    plt.ylabel("Validation acc")
    plt.show()

    model = AutoEncoder(train_matrix.shape[1], best_k)
    (training_acc, valid_acc_list) = train_withoutl2(model, lr, train_matrix, train_data, zero_train_matrix, valid_data, num_epoch)
    epoch_list = np.arange(num_epoch)
    plt.plot(epoch_list, training_acc)
    plt.plot(epoch_list, valid_acc_list)

    plot_legend = ["Training accuracy", "Validation accuracy"]
    plt.legend(plot_legend)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & validation accuracy v/s Epoch(k = {})".format(best_k))
    plt.show()

    model = AutoEncoder(train_matrix.shape[1], 10)
    train_withoutl2(model, lr, train_matrix, train_data, zero_train_matrix, valid_data, num_epoch)
    accuracy = evaluate(model, zero_train_matrix, test_data)
    print("The final test accuracy is {}".format(accuracy))

    print("Now, we include L2 regularisation")
    model = AutoEncoder(train_matrix.shape[1], best_k)
    lamb_set = [0.001, 0.01, 0.1, 1]
    
    for lamb in lamb_set:

        train_withl2(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        val_accuracy = evaluate(model, zero_train_matrix, valid_data)
        test_accuracy = evaluate(model, zero_train_matrix, test_data)
        print("Validation accuracy with lamb = {}: {}".format(lamb, val_accuracy))
        print("Test accuracy with lamb = {}: {}".format(lamb, test_accuracy))


if __name__ == "__main__":
    main()
