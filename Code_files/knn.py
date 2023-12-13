from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, np.transpose(mat))
    print("Validation Accuracy: {}".format(acc))
    return acc

def main():
           
    sparse_matrix = load_train_sparse().toarray()
    val_data = load_valid_csv()
    test_data = load_public_test_csv() 

    # User based collaborative filtering
    # print("User-based Collaborative filtering")
    # k = np.array([1, 6, 11, 16, 21, 26])
    k = np.array([1, 5, 11, 13, 17, 21, 23, 29])
    # accuracy_user = np.zeros(k.shape)

    # for i in range(k.shape[0]):
    #     accuracy_user[i] = knn_impute_by_user(sparse_matrix, val_data, k[i])
    
    # print()
    # k_star_user = k[np.argmax(accuracy_user)]
    # print("The value of k* is {}".format(k_star_user))

    # accuracy_test_user = knn_impute_by_user(sparse_matrix, test_data, k_star_user)
    # print()
    # print("The test accuracy for k* is {}".format(accuracy_test_user))
    # for i in range(k.shape[0]):
    #     print("k = {:2}: Validation accuracy = {}".format(k[i], accuracy_user[i]))
    
    # plt.scatter(k, accuracy_user)
    # plt.plot(k, accuracy_user)
    # plt.title('Accuracy on validation data as function of k')
    # plt.xlabel('k')
    # plt.ylabel('Validation accuracy')
    # plt.show()

    # Item based collaborative filtering
    matrix_transpose = np.transpose(sparse_matrix)
    accuracy_item = np.zeros(k.shape)

    for i in range(k.shape[0]):
        accuracy_item[i] = knn_impute_by_item(matrix_transpose, val_data, k[i])

    print()
    k_star_item = k[np.argmax(accuracy_item)]
    print("The value of k* is {}".format(k_star_item))

    accuracy_test_item = knn_impute_by_item(matrix_transpose, test_data, k_star_item)
    print()
    print("The test accuracy for k* is {}".format(accuracy_test_item))
    
    plt.scatter(k, accuracy_item)
    plt.plot(k, accuracy_item)
    plt.title('Accuracy on validation data as function of k')
    plt.xlabel('k')
    plt.ylabel('Validation accuracy')
    plt.show()
    
if __name__ == "__main__":
    main()
