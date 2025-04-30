import gzip
import numpy as np
import struct
import os

# 数据加载函数
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols)
        return images.astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels].astype(np.float32)

# 加载MNIST数据集
X_train = load_images('data/train-images-idx3-ubyte.gz')
y_train = load_labels('data/train-labels-idx1-ubyte.gz')
y_train_onehot = one_hot_encode(y_train)

X_test = load_images('data/t10k-images-idx3-ubyte.gz')
y_test = load_labels('data/t10k-labels-idx1-ubyte.gz')

# 简单神经网络类
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = 1/(1+np.exp(-self.z1))  
        self.z2 = self.a1.dot(self.W2) + self.b2
        return self.z2
    
    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, X, y_true, y_pred, lr):
        m = X.shape[0]
        grad_z2 = 2*(y_pred - y_true)/m
        dW2 = self.a1.T.dot(grad_z2)
        db2 = np.sum(grad_z2, axis=0)
        grad_a1 = grad_z2.dot(self.W2.T)
        grad_z1 = grad_a1 * self.a1 * (1-self.a1)
        dW1 = X.T.dot(grad_z1)
        db1 = np.sum(grad_z1, axis=0)
        self.W1 -= lr*dW1
        self.b1 -= lr*db1
        self.W2 -= lr*dW2
        self.b2 -= lr*db2

    def save_model(self, save_dir="best_models"):
        """保存模型参数到指定目录"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, 'basic_model.npz'),
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

# 训练函数
def train(model, X, y, y_labels, epochs=10, batch_size=64, lr=0.1):
    num_samples = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss * len(X_batch)
            model.backward(X_batch, y_batch, y_pred, lr)
        epoch_loss /= num_samples
        train_pred = model.predict(X)
        train_acc = np.mean(train_pred == y_labels)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

# 测试函数
def test(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

# 主程序
if __name__ == "__main__":
    model = SimpleNN(784, 128, 10)
    train(model, X_train, y_train_onehot, y_train, epochs=10, lr=0.1)
    model.save_model()  # 训练完成后保存模型
    test(model, X_test, y_test)