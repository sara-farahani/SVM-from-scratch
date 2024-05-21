from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_binary_classification(X_train, y_train, y_pred, class_labels):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    
    plt.figure(figsize=(8, 6))
    for i, target_name in zip([0, 1], class_labels):
        plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], label=target_name)
    plt.title('PCA of Iris dataset (binary classification)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
