import cv2
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 读取yaml文件，加载路径和标签
with open('D:\A_SFA-EYE\dataset\YawDD\datas.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

train_dir = config['train']  # 训练集路径
val_dir = config['val']      # 验证集路径
labels = config['names']     # 类别标签

# 加载图像和标签
X, y = [], []
for label_idx, label in enumerate(labels):
    label_dir = os.path.join(train_dir, label)  # 获取对应类别的图像文件夹
    for file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, file)
        img = cv2.imread(img_path, 0)  # 加载为灰度图像
        img = cv2.resize(img, (64, 64)).flatten()  # 调整为64x64并展平为一维数组
        X.append(img)
        y.append(label_idx)  # 根据类别索引标记标签

X = np.array(X)
y = np.array(y)

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# PCA 降维
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# LDA 进一步降维
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

# SVM 分类
svm = SVC()
svm.fit(X_lda, y_train)
y_pred = svm.predict(X_test_lda)

# 输出准确率
print("准确率：", accuracy_score(y_test, y_pred))
