import gzip
import numpy as np
import struct
import os

# 数据加载函数保持不变
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

class L2RegularizedNN:
    def __init__(self, input_size, hidden_layers, output_size, beta=0.9, l2_lambda=0.001):
        """初始化带L2正则化和动量的神经网络
        Args:
            l2_lambda: L2正则化系数，默认0.001
        """
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
        
        # 输出层
        self.params.append({
            'W': np.random.randn(prev_size, output_size) * 0.01,
            'b': np.zeros(output_size)
        })
        self.velocities.append({
            'W': np.zeros_like(self.params[-1]['W']),
            'b': np.zeros_like(self.params[-1]['b'])
        })
        self.beta = beta

    def forward(self, X):
        self.cache = [X]
        for i in range(len(self.params)-1):
            z = self.cache[-1].dot(self.params[i]['W']) + self.params[i]['b']
            a = 1 / (1 + np.exp(-z))
            self.cache.append(a)
        z_out = self.cache[-1].dot(self.params[-1]['W']) + self.params[-1]['b']
        return z_out

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def compute_loss(self, y_pred, y_true):
        # 计算MSE损失
        mse_loss = np.mean((y_pred - y_true)**2)
        # 计算L2正则化项（仅权重）
        l2_penalty = 0
        for layer in self.params:
            l2_penalty += np.sum(layer['W']**2)
        return mse_loss + 0.5 * self.l2_lambda * l2_penalty  

    def backward(self, X, y_true, y_pred, lr):
        m = X.shape[0]
        grads = []
        
        # 输出层梯度
        dZ = 2*(y_pred - y_true)/m
        dW = self.cache[-1].T.dot(dZ) + self.l2_lambda * self.params[-1]['W']  # 添加L2梯度
        db = np.sum(dZ, axis=0)
        grads.insert(0, (dW, db))
        
        # 反向传播隐藏层
        for i in reversed(range(len(self.params)-1)):
            dA = dZ.dot(self.params[i+1]['W'].T)
            dZ = dA * self.cache[i+1] * (1 - self.cache[i+1])
            dW = self.cache[i].T.dot(dZ) + self.l2_lambda * self.params[i]['W']  # 添加L2梯度
            db = np.sum(dZ, axis=0)
            grads.insert(0, (dW, db))
        
        # 动量更新参数
        for i in range(len(self.params)):
            self.velocities[i]['W'] = self.beta * self.velocities[i]['W'] + (1 - self.beta) * grads[i][0]
            self.velocities[i]['b'] = self.beta * self.velocities[i]['b'] + (1 - self.beta) * grads[i][1]
            
            self.params[i]['W'] -= lr * self.velocities[i]['W']
            self.params[i]['b'] -= lr * self.velocities[i]['b']

    def save_model(self, save_dir="best_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, 'q3_model.npz'),
                 params=self.params,
                 velocities=self.velocities,
                 l2_lambda=self.l2_lambda)

# 训练函数保持不变
def train(model, X, y, y_labels, epochs=15, batch_size=128, lr=0.05):
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

# 测试函数保持不变
def test(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 初始化带L2正则化的模型（λ=0.001）
    model = L2RegularizedNN(784, [256, 128], 10, beta=0.9, l2_lambda=0.001)
    
    # 训练模型
    train(model, X_train, y_train_onehot, y_train)
    
    # 保存模型
    model.save_model()
    
    # 测试模型
    test(model, X_test, y_test)