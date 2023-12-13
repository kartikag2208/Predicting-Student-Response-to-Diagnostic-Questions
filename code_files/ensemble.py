from utils import *
import numpy as np
import matplotlib.pyplot as plt

N = 542
D = 1774

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
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

def update_theta_beta(data, train_matrix, lr, theta, beta):

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

def irt(data, train_matrix, val_data, lr, iterations):

    theta = np.zeros(N)
    beta = np.zeros(D)

    val_acc_lst = []
    training_neg_lld = []
    validation_neg_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, beta)
        score = evaluate(val_data, theta, beta)
        val_acc_lst.append(score)
        training_neg_lld.append(neg_lld)
        validation_neg_lld.append(neg_log_likelihood(val_data, theta, beta))

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, train_matrix, lr, theta, beta)

    # fig, ax = plt.subplots()
    # ax.plot(np.arange(iterations), training_neg_lld)
    # ax.plot(np.arange(iterations), validation_neg_lld)
    # ax.xaxis.set_label_text('Iterations')
    # ax.yaxis.set_label_text('Negative log-likelihoods')
    # ax.set_title('Training and valid neg log-likelihoods v/s Iterations')
    # ax.legend(['Train_neg_lld', 'Valid_neg_llds'])
    # plt.savefig('./2b. train and val negtive log likelihoods for each iteration.png')

    # plt.close(fig)
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

    lr = 0.003
    iterations = 200

    # Number of base models
    num_base_models = 3
    base_model_thetas = []
    base_model_betas = []

    for base_model_index in range(num_base_models):
        
        # Bootstrap sampling
        bootstrap_indices = np.random.choice(len(train_data["user_id"]), len(train_data["user_id"]), replace=True)
        bootstrap_data = {
            "user_id": [],
            "question_id": [],
            "is_correct": []
        }

        for i in range(len(bootstrap_indices)):
            bootstrap_data["user_id"].append(train_data["user_id"][bootstrap_indices[i]])
            bootstrap_data["question_id"].append(train_data["question_id"][bootstrap_indices[i]])
            bootstrap_data["is_correct"].append(train_data["is_correct"][bootstrap_indices[i]])
        
        train_matrix = np.empty((N, D))
        train_matrix[:] = np.nan
        
        for i in range(len(train_data["user_id"])):
            i_index = bootstrap_data["user_id"][i]
            j_index = bootstrap_data["question_id"][i]
            if(bootstrap_data["is_correct"][i] == 1):
                train_matrix[i_index][j_index] = 1
            if(bootstrap_data["is_correct"][i] == 0):
                train_matrix[i_index][j_index] = 0

        # Train base IRT model
        theta, beta = irt(bootstrap_data, train_matrix, val_data, lr, iterations)

        # Save base model parameters
        base_model_thetas.append(theta)
        base_model_betas.append(beta)
        print("Model {} completed\n".format(base_model_index))

    # Aggregate predictions of base models (majority voting)
    aggregated_predictions = np.zeros(len(val_data["is_correct"]))
    aggregated_predictions_test = np.zeros(len(test_data["is_correct"]))

    for base_model_index in range(num_base_models):
        theta = base_model_thetas[base_model_index]
        beta = base_model_betas[base_model_index]

        # Make predictions using the current base model
        predictions = [sigmoid(theta[u] - beta[q]) >= 0.5 for u, q in zip(val_data["user_id"], val_data["question_id"])]
        predictions_test = [sigmoid(theta[u] - beta[q]) >= 0.5 for u, q in zip(test_data["user_id"], test_data["question_id"])]
        # Add the predictions to the aggregated predictions
        aggregated_predictions += predictions
        aggregated_predictions_test += predictions_test

    # Convert aggregated predictions to binary (majority voting)
    aggregated_predictions = (aggregated_predictions >= (num_base_models / 2)).astype(int)
    aggregated_predictions_test = (aggregated_predictions_test >= (num_base_models / 2)).astype(int)


    # Evaluate aggregated predictions
    aggregated_accuracy = np.sum(aggregated_predictions == val_data["is_correct"]) / len(val_data["is_correct"])
    aggregated_accuracy_test = np.sum(aggregated_predictions_test == test_data["is_correct"]) / len(test_data["is_correct"])
    print("Aggregated Model Accuracy:", aggregated_accuracy)
    print("Aggregated Model Accuracy Test:", aggregated_accuracy_test)

if __name__ == "__main__":
    main()
