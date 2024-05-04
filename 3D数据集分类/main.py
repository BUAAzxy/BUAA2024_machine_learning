import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons')
plt.show()

# 生成额外的500个样本数据集
X_test, labels_test = make_moons_3d(n_samples=500, noise=0.2)

# 区分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# 逻辑回归
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(labels_test, lr_pred)

# 支持向量机
svm_kernels = ['linear', 'poly', 'rbf']
svm_models = []
for kernel in svm_kernels:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    svm_models.append(svm_model)

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(labels_test, xgb_pred)

# 绘图部分
def plot_decision_boundary(ax, model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')

fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# Logistic Regression
plot_decision_boundary(axs[0], lr_model, X_test[:, :2], labels_test)
axs[0].set_title('Logistic Regression')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# SVMs
for i, (model, kernel) in enumerate(zip(svm_models, svm_kernels), start=1):
    plot_decision_boundary(axs[i], model, X_test[:, :2], labels_test)
    axs[i].set_title(f'SVM ({kernel.capitalize()} Kernel)')
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
    
# XGBoost
plot_decision_boundary(axs[4], xgb_model, X_test[:, :2], labels_test)
axs[4].set_title('XGBoost')
axs[4].set_xlabel('X')
axs[4].set_ylabel('Y')

plt.tight_layout()
plt.show()

# Print accuracies
print("Logistic Regression Accuracy:", lr_accuracy)
print("SVM Accuracies with different kernels:", [accuracy_score(labels_test, model.predict(X_test)) for model in svm_models])
print("XGBoost Accuracy:", xgb_accuracy)
