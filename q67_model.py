import gzip
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# 数据加载函数（保持图像结构）
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images.astype(np.float32) / 255.0  # 形状 (N, 28, 28)

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels].astype(np.float32)

# 数据增强函数（第六点）
def augment_image(img):
    """对单张28x28图像进行随机变换"""
    # 随机平移（-2到2像素）
    dx, dy = np.random.randint(-2, 3, 2)
    # 随机旋转（-15到15度）
    angle = np.random.uniform(-15, 15)
    # 随机缩放（0.9到1.1倍）
    scale = np.random.uniform(0.9, 1.1)
    
    # 构建变换矩阵
    transform = np.array([
        [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), dx],
        [scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), dy]
    ])
    
    # 应用仿射变换
    return affine_transform(img, transform, output_shape=(28, 28), mode='nearest')

# 加载MNIST数据集
X_train_orig = load_images('data/train-images-idx3-ubyte.gz')  # 形状 (60000, 28, 28)
y_train = load_labels('data/train-labels-idx1-ubyte.gz')
y_train_onehot = one_hot_encode(y_train)

X_test_orig = load_images('data/t10k-images-idx3-ubyte.gz')    # 形状 (10000, 28, 28)
y_test = load_labels('data/t10k-labels-idx1-ubyte.gz')

class EnhancedNN:
    def __init__(self, input_size, hidden_layers, output_size, beta=0.9, l2_lambda=0.001):
        """初始化带动量、L2正则化的神经网络"""
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

    def _softmax(self, z):
        # 数值稳定实现
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def forward(self, X):
        self.cache = [X]
        for i in range(len(self.params)-1):
            z = self.cache[-1].dot(self.params[i]['W']) + self.params[i]['b']
            a = 1 / (1 + np.exp(-z))  # Sigmoid激活
            self.cache.append(a)
        z_out = self.cache[-1].dot(self.params[-1]['W']) + self.params[-1]['b']
        return z_out

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def compute_loss(self, y_pred_logits, y_true):
        # 交叉熵损失 + L2正则化
        m = y_true.shape[0]
        probs = self._softmax(y_pred_logits)
        log_probs = -np.log(probs[np.arange(m), np.argmax(y_true, axis=1)] + 1e-8)
        ce_loss = np.mean(log_probs)
        
        l2_penalty = sum(np.sum(layer['W']**2) for layer in self.params)
        return ce_loss + 0.5 * self.l2_lambda * l2_penalty

    def backward(self, X, y_true, y_pred_logits, lr):
        m = X.shape[0]
        grads = []
        
        # 交叉熵梯度计算
        probs = self._softmax(y_pred_logits)
        dZ = (probs - y_true) / m
        
        # 反向传播
        for i in reversed(range(len(self.params))):
            # 计算梯度（含L2正则）
            dW = self.cache[i].T.dot(dZ) + self.l2_lambda * self.params[i]['W']
            db = np.sum(dZ, axis=0)
            grads.insert(0, (dW, db))
            
            if i > 0:
                dZ = dZ.dot(self.params[i]['W'].T) * self.cache[i] * (1 - self.cache[i])
        
        # 动量更新
        for i in range(len(self.params)):
            self.velocities[i]['W'] = self.beta * self.velocities[i]['W'] + (1 - self.beta) * grads[i][0]
            self.velocities[i]['b'] = self.beta * self.velocities[i]['b'] + (1 - self.beta) * grads[i][1]
            self.params[i]['W'] -= lr * self.velocities[i]['W']
            self.params[i]['b'] -= lr * self.velocities[i]['b']

    def visualize_weights(self, layer_idx=0, save_path='weight_visualization.png'):
        """第七点：权重可视化"""
        weights = self.params[layer_idx]['W']
        n_features = weights.shape[1]
        
        plt.figure(figsize=(15, 8))
        for i in range(min(n_features, 128)):  # 最多显示128个特征
            plt.subplot(8, 16, i+1)
            plt.imshow(weights[:, i].reshape(28, 28), cmap='gray', vmin=-0.5, vmax=0.5)
            plt.axis('off')
        plt.suptitle(f'First Hidden Layer Weight Patterns (λ={self.l2_lambda})')
        plt.savefig(save_path)
        plt.show()

    def save_model(self, save_dir="best_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, 'q67_model.npz'),
                 params=self.params,
                 velocities=self.velocities,
                 l2_lambda=self.l2_lambda)

# 训练函数（含第六点数据增强）
def train(model, X_orig, y, y_labels, epochs=20, batch_size=256, lr=0.03):
    num_samples = X_orig.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, batch_size):
            # 数据增强
            X_batch = np.array([augment_image(img) for img in X_orig[indices[i:i+batch_size]]])
            X_batch = X_batch.reshape(-1, 28*28)  # 展平输入
            
            y_batch = y[indices[i:i+batch_size]]
            
            # 前向传播
            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch)
            
            # 反向传播
            model.backward(X_batch, y_batch, logits, lr)
            
            epoch_loss += loss * len(X_batch)
        
        epoch_loss /= num_samples
        # 在原始数据上计算准确率
        train_pred = model.predict(X_orig.reshape(-1, 28*28))
        train_acc = np.mean(train_pred == y_labels)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

# 测试函数
def test(model, X_orig, y):
    X_flatten = X_orig.reshape(-1, 28*28)
    predictions = model.predict(X_flatten)
    accuracy = np.mean(predictions == y)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 初始化模型
    model = EnhancedNN(784, [256, 128], 10, beta=0.9, l2_lambda=0.001)
    
    # 训练（使用数据增强）
    train(model, X_train_orig, y_train_onehot, y_train, epochs=20, lr=0.25)
    
    # 可视化第一层权重
    model.visualize_weights(layer_idx=0)
    
    # 保存模型
    model.save_model()
    
    # 测试
    test(model, X_test_orig, y_test)