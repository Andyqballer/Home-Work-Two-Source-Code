import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
housing_dataset = '/content/Housing.csv'
data = pd.read_csv(housing_dataset)

# Encode categorical variables
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0})

# Split dataset
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    losses = []
    
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1/(2*m)) * np.sum(errors ** 2)
        losses.append(loss)
        gradients = (1/m) * X.T.dot(errors)
        theta -= learning_rate * gradients
        
    return theta, losses

# Problem 1.a: Using selected features
features_a = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X_train_a = X_train[:, [X.columns.get_loc(f) for f in features_a]]
X_test_a = X_test[:, [X.columns.get_loc(f) for f in features_a]]

theta_a, losses_a = gradient_descent(X_train_a, y_train.values, learning_rate=0.01, iterations=1000)

# Validation Loss
val_predictions_a = X_test_a.dot(theta_a)
val_loss_a = np.mean((val_predictions_a - y_test) ** 2)

print(f"Validation Loss for Model A: {val_loss_a}")

plt.plot(losses_a, label='Training Loss Model A')
plt.title('Training Loss for Model A')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Problem 1.b: Using all features
features_b = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 
              'hotwaterheating', 'airconditioning', 'parking', 'prefarea']
X_train_b = X_train[:, [X.columns.get_loc(f) for f in features_b]]
X_test_b = X_test[:, [X.columns.get_loc(f) for f in features_b]]

theta_b, losses_b = gradient_descent(X_train_b, y_train.values, learning_rate=0.01, iterations=1000)

# Validation Loss
val_predictions_b = X_test_b.dot(theta_b)
val_loss_b = np.mean((val_predictions_b - y_test) ** 2)

print(f"Validation Loss for Model B: {val_loss_b}")

plt.plot(losses_b, label='Training Loss Model B', color='orange')
plt.title('Training Loss for Model B')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Problem 2

# Problem 2.a: Normalization and Standardization for Model A
scaler_norm = MinMaxScaler()
X_train_norm_a = scaler_norm.fit_transform(X_train_a)
X_test_norm_a = scaler_norm.transform(X_test_a)

theta_norm_a, losses_norm_a = gradient_descent(X_train_norm_a, y_train.values)

scaler_std = StandardScaler()
X_train_std_a = scaler_std.fit_transform(X_train_a)
X_test_std_a = scaler_std.transform(X_test_a)

theta_std_a, losses_std_a = gradient_descent(X_train_std_a, y_train.values)

# Validation Loss
val_predictions_norm_a = X_test_norm_a.dot(theta_norm_a)
val_loss_norm_a = np.mean((val_predictions_norm_a - y_test) ** 2)

val_predictions_std_a = X_test_std_a.dot(theta_std_a)
val_loss_std_a = np.mean((val_predictions_std_a - y_test) ** 2)

plt.plot(losses_norm_a, label='Training Loss Normalization')
plt.plot(losses_std_a, label='Training Loss Standardization', color='orange')
plt.title('Training Loss for Normalization vs Standardization (Model A)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print Training Losses for Model A
print(f"Final Training Loss (Normalization Model A): {losses_norm_a[-1]}")
print(f"Final Training Loss (Standardization Model A): {losses_std_a[-1]}")

# Problem 2.b: Normalization and Standardization for Model B
scaler_norm_b = MinMaxScaler()
X_train_norm_b = scaler_norm_b.fit_transform(X_train_b)
X_test_norm_b = scaler_norm_b.transform(X_test_b)

theta_norm_b, losses_norm_b = gradient_descent(X_train_norm_b, y_train.values)

scaler_std_b = StandardScaler()
X_train_std_b = scaler_std_b.fit_transform(X_train_b)
X_test_std_b = scaler_std_b.transform(X_test_b)

theta_std_b, losses_std_b = gradient_descent(X_train_std_b, y_train.values)

# Validation Loss
val_predictions_norm_b = X_test_norm_b.dot(theta_norm_b)
val_loss_norm_b = np.mean((val_predictions_norm_b - y_test) ** 2)

val_predictions_std_b = X_test_std_b.dot(theta_std_b)
val_loss_std_b = np.mean((val_predictions_std_b - y_test) ** 2)

plt.plot(losses_norm_b, label='Training Loss Normalization (Model B)')
plt.plot(losses_std_b, label='Training Loss Standardization (Model B)', color='orange')
plt.title('Training Loss for Normalization vs Standardization (Model B)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print Training Losses for Model B
print(f"Final Training Loss (Normalization Model B): {losses_norm_b[-1]}")
print(f"Final Training Loss (Standardization Model B): {losses_std_b[-1]}")


# problem 3

# Gradient Descent with Penalty function
def gradient_descent_with_penalty(X, y, learning_rate=0.01, iterations=1000, penalty=1):
    m = len(y)
    theta = np.zeros(X.shape[1])
    losses = []
    
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1/(2*m)) * (np.sum(errors ** 2) + penalty * np.sum(theta ** 2))
        losses.append(loss)
        gradients = (1/m) * X.T.dot(errors) + (penalty/m) * theta
        theta -= learning_rate * gradients
        
    return theta, losses

# Problem 3.a: Add penalty to loss function for normalization
theta_norm_a_penalty, losses_norm_a_penalty = gradient_descent_with_penalty(X_train_norm_a, y_train.values)

plt.plot(losses_norm_a_penalty, label='Training Loss Normalization with Penalty')
plt.title('Training Loss for Normalization with Penalty (Model A)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print Training Loss for Model A with Normalization
print(f"Final Training Loss (Normalization with Penalty Model A): {losses_norm_a_penalty[-1]}")

# Problem 3.b: Add penalty to loss function for standardization
theta_std_b_penalty, losses_std_b_penalty = gradient_descent_with_penalty(X_train_std_b, y_train.values)

plt.plot(losses_std_b_penalty, label='Training Loss Standardization with Penalty (Model B)', color='orange')
plt.title('Training Loss for Standardization with Penalty (Model B)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print Training Loss for Model B with Standardization
print(f"Final Training Loss (Standardization with Penalty Model B): {losses_std_b_penalty[-1]}")
