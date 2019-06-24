"""
    Main file for testing neural network functionality.
"""

import network as n

(train_d, train_l), (test_d, test_l) = n.MNIST_normalized()

my_model = n.get_trained_network(train_d, train_l, test_d, test_l, 1)

correct = 0
for i in range(test_l.shape[0]):
#    print("Test example: ", i,". Test value: ", n.np.argmax(test_l[i]))
#    print("Function 'single_prediction' results: ", n.single_prediction(test_d[i], my_model))
    if n.np.argmax(test_l[i]) == n.single_prediction(test_d[i], my_model):
        correct += 1

print("Accuracy of 'single prediction' function: ", correct / test_l.shape[0])