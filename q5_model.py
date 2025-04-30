import gzip
import numpy as np
import struct
import os

# 数据加载函数
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
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

# 卷积层实现
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.kernel.shape
        H_out = (H + 2*self.padding - HH) // self.stride + 1
        W_out = (W + 2*self.padding - WW) // self.stride + 1
        
        x_pad = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 
                        mode='constant')
        out = np.zeros((N, F, H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + HH
                w_start = j * self.stride
                w_end = w_start + WW
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.tensordot(x_slice, self.kernel, axes=([1,2,3], [1,2,3])) + self.bias
        self.cache = (x, x_pad)
        return out

    def backward(self, dout, lr):
        x, x_pad = self.cache
        N, C, H, W = x.shape
        F, _, HH, WW = self.kernel.shape
        
        dx_pad = np.zeros_like(x_pad)
        dW = np.zeros_like(self.kernel)
        db = np.sum(dout, axis=(0,2,3))
        
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                h_start = i * self.stride
                h_end = h_start + HH
                w_start = j * self.stride
                w_end = w_start + WW
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                dW += np.tensordot(dout[:, :, i, j], x_slice, axes=([0], [0]))
                dx_pad[:, :, h_start:h_end, w_start:w_end] += np.tensordot(dout[:, :, i, j], self.kernel, axes=([1], [0]))
        
        dx = dx_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        self.kernel -= lr * dW / N
        self.bias -= lr * db / N
        return dx

# 最大池化层实现
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        mask = np.zeros_like(x)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(x_slice, axis=(2,3))
                mask[:, :, h_start:h_end, w_start:w_end] = (x_slice == out[:, :, i, j, None, None])
        self.cache = mask
        return out

    def backward(self, dout):
        mask = self.cache
        dx = np.zeros_like(mask)
        N, C, H, W = mask.shape
        
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                dx[:, :, h_start:h_end, w_start:w_end] += dout[:, :, i, j, None, None] * mask[:, :, h_start:h_end, w_start:w_end]
        return dx

# 完整的CNN实现
class CNN:
    def __init__(self):
        # 卷积层
        self.conv1 = Conv2D(1, 16, kernel_size=3, padding=1)
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
        self.pool2 = MaxPool2D()
        
        # 全连接层（修正形状）
        self.fc1 = np.random.randn(32*7*7, 128) * np.sqrt(2/(32*7*7))
        self.b1 = np.zeros(128)
        self.fc2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)
        
        # 动量参数
        self.velocity = {
            'fc1': np.zeros_like(self.fc1),
            'b1': np.zeros_like(self.b1),
            'fc2': np.zeros_like(self.fc2),
            'b2': np.zeros_like(self.b2)
        }
        self.beta = 0.9

    def forward(self, x):
        # 前向传播流程
        x = x[:, np.newaxis, :, :]  # 添加通道维度
        
        # 第一卷积层
        x = self.conv1.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool1.forward(x)
        
        # 第二卷积层
        x = self.conv2.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool2.forward(x)
        
        # 展平
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # 全连接层
        self.fc1_input = x_flat  # 记录展平后的输入用于反向传播
        x = x_flat.dot(self.fc1) + self.b1
        x = np.maximum(0, x)  # ReLU
        self.fc1_activation = x  # 记录激活值
        x = x.dot(self.fc2) + self.b2
        
        return x

    def backward(self, dout, lr):
        batch_size = dout.shape[0]
        
        # 全连接层反向传播
        d_fc2 = self.fc1_activation.T.dot(dout)  # (128,10)
        d_b2 = np.sum(dout, axis=0)
        
        dx = dout.dot(self.fc2.T)  # (batch,128)
        dx = (dx > 0).astype(float) * dx  # ReLU梯度
        
        d_fc1 = self.fc1_input.T.dot(dx)  # (1568,128)
        d_b1 = np.sum(dx, axis=0)
        
        # 动量更新
        self.velocity['fc2'] = self.beta * self.velocity['fc2'] + (1 - self.beta) * d_fc2
        self.velocity['b2'] = self.beta * self.velocity['b2'] + (1 - self.beta) * d_b2
        self.velocity['fc1'] = self.beta * self.velocity['fc1'] + (1 - self.beta) * d_fc1
        self.velocity['b1'] = self.beta * self.velocity['b1'] + (1 - self.beta) * d_b1
        
        # 参数更新
        self.fc2 -= lr * self.velocity['fc2'] / batch_size
        self.b2 -= lr * self.velocity['b2'] / batch_size
        self.fc1 -= lr * self.velocity['fc1'] / batch_size
        self.b1 -= lr * self.velocity['b1'] / batch_size
        
        # 传播到卷积层
        dx = dx.dot(self.fc1.T).reshape(batch_size, 32, 7, 7)
        return dx

    def save_model(self, save_dir="best_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, 'q5_model.npz'),
                 conv1_kernel=self.conv1.kernel,
                 conv1_bias=self.conv1.bias,
                 conv2_kernel=self.conv2.kernel,
                 conv2_bias=self.conv2.bias,
                 fc1=self.fc1,
                 b1=self.b1,
                 fc2=self.fc2,
                 b2=self.b2)

# 训练函数
def train(model, X, y, y_labels, epochs=10, batch_size=64, lr=0.01):
    num_samples = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # 前向传播
            logits = model.forward(X_batch)
            
            # 计算损失
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            loss = -np.mean(np.log(probs[np.arange(len(y_batch)), np.argmax(y_batch, axis=1)] + 1e-8))
            epoch_loss += loss * len(X_batch)
            
            # 计算梯度
            grad = probs.copy()
            grad[np.arange(len(y_batch)), np.argmax(y_batch, axis=1)] -= 1
            grad /= len(y_batch)
            
            # 反向传播
            model.backward(grad, lr)
        
        # 计算准确率
        epoch_loss /= num_samples
        train_pred = np.argmax(model.forward(X), axis=1)
        train_acc = np.mean(train_pred == y_labels)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

# 测试函数
def test(model, X, y):
    logits = model.forward(X)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    # 初始化模型
    model = CNN()
    
    # 训练模型
    train(model, X_train, y_train_onehot, y_train, epochs=10, lr=0.01)
    
    # 保存模型
    model.save_model()
    
    # 测试模型
    test(model, X_test, y_test)