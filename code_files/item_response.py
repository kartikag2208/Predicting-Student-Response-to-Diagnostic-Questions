from utils import *
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    m, n = matrix.shape
    log_lklihood = 0.
    for i in range(m):
        for j in range(n):
            if((matrix[i][j] == 1) or (matrix[i][j] == 0)):
                c = matrix[i][j]
                log_lklihood = log_lklihood + (c*np.log(sigmoid(theta[i] - beta[j])) + (1-c)*np.log(1 - sigmoid(theta[i] - beta[j])))

    return -log_lklihood

def neg_lld_validation(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]

        # prob = p(c_uq|theta_u, beta_q)
        x = theta[u] - beta[q]
        prob = sigmoid(x)

        c = data["is_correct"][i]
        log_lklihood += c * np.log(prob) + (1 - c) * np.log((1 - prob))
    return -log_lklihood

def update_theta_beta(train_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """    
    N, D = train_matrix.shape
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)

    nan_positions = np.isnan(train_matrix)
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[nan_positions] = 0

    # update theta
    x = theta_mat - beta_mat
    prob = sigmoid(x)
    prob[nan_positions] = 0
    theta = theta - lr*np.sum(prob - zero_train_matrix, axis=1)

    # update beta
    x = theta_mat - beta_mat
    prob = sigmoid(x) 
    prob[nan_positions] = 0
    beta = beta - lr*np.sum(zero_train_matrix - prob, axis=0)

    return theta, beta


def irt(train_matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    N, D = train_matrix.shape
    theta = np.zeros(N)
    beta = np.zeros(D)

    val_acc_lst = []
    training_neg_lld = []
    validation_neg_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(train_matrix, theta, beta)
        score = evaluate(val_data, theta, beta)
        val_acc_lst.append(score)
        training_neg_lld.append(neg_lld)
        validation_neg_lld.append(neg_lld_validation(val_data, theta, beta))

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(train_matrix, lr, theta, beta)

    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), training_neg_lld)
    ax.plot(np.arange(iterations), validation_neg_lld)
    ax.xaxis.set_label_text('Iterations')
    ax.yaxis.set_label_text('Negative log-likelihoods')
    ax.set_title('Training and valid neg log-likelihoods v/s Iterations')
    ax.legend(['Train_neg_lld', 'Valid_neg_llds'])
    plt.savefig('./2b. train and val negtive log likelihoods for each iteration.png')

    plt.close(fig)
    return theta, beta


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5) 
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv()
    sparse_matrix = load_train_sparse().toarray()
    val_data = load_valid_csv()
    test_data = load_public_test_csv()

    N, D = sparse_matrix.shape
    lr = 0.003
    iterations = 200
    theta, beta = irt(sparse_matrix, val_data, lr, iterations)

    valid_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)

    print(valid_accuracy)
    print(test_accuracy)

    question_list = np.random.choice(D, 5, replace=False)
    theta_axis = np.linspace(-5, 5, 101)

    colors = ['r', 'g', 'b', 'm', 'y']

    for i in range(5):
        
        x = theta_axis - beta[question_list[i]]
        prob = sigmoid(x)
        plt.plot(theta_axis, prob, color=colors[i])

    plot_legend = []
    for i in range(5):
        plot_legend.append("question_id: {}".format(question_list[i]))
    
    plt.legend(plot_legend) 
    plt.xlabel("theta")
    plt.ylabel("p(c_ij)")
    plt.title("p(c_ij) v/s theta for 5 different questions")       
    plt.show()

if __name__ == "__main__":
    main()
