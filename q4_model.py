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

class SoftmaxCrossEntropyNN:
    def __init__(self, input_size, hidden_layers, output_size, beta=0.9, l2_lambda=0.001):
        """带Softmax和交叉熵损失的神经网络"""
        self.params = []
        self.velocities = []
        self.l2_lambda = l2_lambda
        prev_size = input_size
        
        # 初始化隐藏层
        for hidden_size in hidden_layers:
            self.params.append({
                'W': np.random.randn(prev_size, hidden_size) * np.sqrt(2/prev_size),
                'b': np.zeros(hidden_size)
            })
            self.velocities.append({
                'W': np.zeros_like(self.params[-1]['W']),
                'b': np.zeros_like(self.params[-1]['b'])
            })
            prev_size = hidden_size
        
        # 输出层（最后一层为Softmax输入）
        self.params.append({
            'W': np.random.randn(prev_size, output_size) * 0.01,
            'b': np.zeros(output_size)
        })
        self.velocities.append({
            'W': np.zeros_like(self.params[-1]['W']),
            'b': np.zeros_like(self.params[-1]['b'])
        })
        self.beta = beta

    def _softmax(self, z):
        # 数值稳定实现
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def forward(self, X):
        self.cache = [X]
        for i in range(len(self.params)-1):
            z = self.cache[-1].dot(self.params[i]['W']) + self.params[i]['b']
            a = 1 / (1 + np.exp(-z))  # 隐藏层仍使用Sigmoid
            self.cache.append(a)
        
        # 输出层计算（无激活，Softmax在损失函数中处理）
        z_out = self.cache[-1].dot(self.params[-1]['W']) + self.params[-1]['b']
        return z_out

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def compute_loss(self, y_pred_logits, y_true):
        # 计算交叉熵损失
        m = y_true.shape[0]
        probs = self._softmax(y_pred_logits)
        log_probs = -np.log(probs[np.arange(m), np.argmax(y_true, axis=1)] + 1e-8)  # 防止log(0)
        ce_loss = np.mean(log_probs)
        
        # L2正则化项
        l2_penalty = 0
        for layer in self.params:
            l2_penalty += np.sum(layer['W']**2)
        return ce_loss + 0.5 * self.l2_lambda * l2_penalty

    def backward(self, X, y_true, y_pred_logits, lr):
        m = X.shape[0]
        grads = []
        
        # 交叉熵梯度计算
        probs = self._softmax(y_pred_logits)
        dZ = (probs - y_true) / m  # 核心简化公式
        
        # 输出层梯度（含L2）
        dW = self.cache[-1].T.dot(dZ) + self.l2_lambda * self.params[-1]['W']
        db = np.sum(dZ, axis=0)
        grads.insert(0, (dW, db))
        
        # 反向传播隐藏层
        for i in reversed(range(len(self.params)-1)):
            dA = dZ.dot(self.params[i+1]['W'].T)
            dZ = dA * self.cache[i+1] * (1 - self.cache[i+1])  # Sigmoid导数
            dW = self.cache[i].T.dot(dZ) + self.l2_lambda * self.params[i]['W']
            db = np.sum(dZ, axis=0)
            grads.insert(0, (dW, db))
        
        # 动量更新
        for i in range(len(self.params)):
            self.velocities[i]['W'] = self.beta * self.velocities[i]['W'] + (1 - self.beta) * grads[i][0]
            self.velocities[i]['b'] = self.beta * self.velocities[i]['b'] + (1 - self.beta) * grads[i][1]
            
            self.params[i]['W'] -= lr * self.velocities[i]['W']
            self.params[i]['b'] -= lr * self.velocities[i]['b']

    def save_model(self, save_dir="best_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, 'q4_model.npz'),
                 params=self.params,
                 velocities=self.velocities,
                 l2_lambda=self.l2_lambda)

# 训练函数
def train(model, X, y, y_labels, epochs=15, batch_size=128, lr=0.05):
    num_samples = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            y_pred_logits = model.forward(X_batch)
            loss = model.compute_loss(y_pred_logits, y_batch)
            epoch_loss += loss * len(X_batch)
            
            model.backward(X_batch, y_batch, y_pred_logits, lr)
        
        epoch_loss /= num_samples
        train_pred = model.predict(X)
        train_acc = np.mean(train_pred == y_labels)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

# 测试函数保持不变
def test(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 初始化带Softmax和交叉熵的模型
    model = SoftmaxCrossEntropyNN(784, [256, 128], 10, beta=0.9, l2_lambda=0.001)
    
    # 训练模型
    train(model, X_train, y_train_onehot, y_train)
    
    # 保存模型
    model.save_model()
    
    # 测试模型
    test(model, X_test, y_test)