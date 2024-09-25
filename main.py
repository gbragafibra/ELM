import numpy as np
import matplotlib.pyplot as plt
from utils import *
from ELM import *

def run(n_sims, X, y, save = False):
    # Train and test accuracies
    train_accs = []
    test_accs = []
    
    #hidden dimension layer range
    hid_dim_range = np.arange(10, 300, 50)
    for hid_dim in hid_dim_range:
        # train and test accs for a specific hid_dim
        train_accuracy = [] 
        test_accuracy = []
        print(f"With hidden dimension {hid_dim}.")
        for sim in range(n_sims):
            print(f"Simulation {sim + 1}/{n_sims}")
            X_train, y_train, X_test, y_test = train_test_split(X, y)
            y_train_one_hot = one_hot_encode(y_train)
            y_test_one_hot = one_hot_encode(y_test)

            net = ELM(X_train, y_train_one_hot, hid_dim) #init
            net.train() #train
            train_acc, test_acc = net.get_accuracy(X_train, X_test, y_train, y_test)
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            
        train_accs.append((np.mean(train_accuracy), np.std(train_accuracy)))
        test_accs.append((np.mean(test_accuracy), np.std(test_accuracy)))
    train_μ, train_σ = zip(*train_accs)
    test_μ, test_σ = zip(*test_accs)

    print("Finished!")

    plt.errorbar(hid_dim_range, train_μ, yerr=train_σ, fmt=".", capsize=5, capthick=1, ecolor="r", label="Train acc")
    plt.errorbar(hid_dim_range, test_μ, yerr=test_σ, fmt=".", capsize=5, capthick=1, ecolor="k", label="Test acc")
    plt.xlabel("Hidden dim")
    plt.ylabel("Accuracy")
    plt.legend()

    if save:
        plt.savefig("performance.png", format="png", dpi=300)
    
    plt.show()


if __name__ == "__main__":
	X, y = MNIST_load()
	run(10, X, y, save=True)
