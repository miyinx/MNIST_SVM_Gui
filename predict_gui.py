import sys
import os
import numpy as np
import joblib
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def preprocess_image(path):
    image = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(image, dtype=np.uint8)
    if arr.mean() > 127:
        arr = 255 - arr
    data = np.round(arr / 255.0).reshape(1, -1)
    return data, arr

def predict_with_confidence(clf, data):
    pred = clf.predict(data)
    label = pred[0]
    try:
        classes = list(clf.classes_)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(data)[0]
        else:
            scores = clf.decision_function(data)
            if scores.ndim == 1:
                scores = np.expand_dims(scores, 0)
            s = scores[0]
            s = s - np.max(s)
            e = np.exp(s)
            probs = e / e.sum()
        idx_map = {c: i for i, c in enumerate(classes)}
        ci = idx_map[label]
        conf = float(probs[ci])
        top_idx = np.argsort(probs)[::-1][:3]
        top_list = [(str(classes[i]), float(probs[i])) for i in top_idx]
        return str(label), conf, top_list, classes, probs
    except Exception:
        return str(label), None, [], [], None

class PredictWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST SVM 识别")
        self.resize(1000, 700)
        self.model = None
        self.image_path = None

        self.btn_select = QPushButton("选择图片")
        self.btn_select.clicked.connect(self.on_select)
        self.btn_random = QPushButton("随机测试")
        self.btn_random.clicked.connect(self.on_random)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(420)

        self.canvas = FigureCanvas(Figure(figsize=(4.5, 4.5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.setMinimumSize(420, 420)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.result = QLabel("结果: ")
        self.result.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.preview, 1)
        row.addWidget(self.canvas, 1)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_select)
        btns.addWidget(self.btn_random)
        layout.addLayout(btns)
        layout.addLayout(row)
        layout.addWidget(self.result)
        self.setLayout(layout)

        self.load_model()

    def load_model(self):
        model_path = os.path.join(sys.path[0], "svm.model")
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", "模型文件不存在: svm.model")
            self.btn_select.setEnabled(False)
            return
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
            self.btn_select.setEnabled(False)

    def on_select(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(self.preview.width(), self.preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        data, arr = preprocess_image(path)
        label, conf, top, classes, probs = predict_with_confidence(self.model, data)
        if conf is None:
            self.result.setText(f"结果: {label}")
        else:
            top_str = ", ".join([f"{k}:{v:.4f}" for k, v in top])
            self.result.setText(f"结果: {label}  置信度: {conf:.4f}  Top-3: {top_str}")
        if probs is not None:
            self.ax.clear()
            n = len(probs)
            top_idx = list(np.argsort(probs)[::-1][:3])
            labels = [''] * n
            for i in top_idx:
                labels[i] = str(classes[i])
            colors = ['lightgray'] * n
            base_colors = ['tab:blue', 'tab:orange', 'tab:green']
            for j, i in enumerate(top_idx):
                colors[i] = base_colors[j % len(base_colors)]
            wedges, texts, autotexts = self.ax.pie(probs, labels=labels, colors=colors, autopct='%1.2f', startangle=90)
            for i in range(n):
                if i not in top_idx:
                    if i < len(texts):
                        texts[i].set_visible(False)
                    if i < len(autotexts):
                        autotexts[i].set_visible(False)
            self.ax.axis('equal')
            self.canvas.draw()

    def on_random(self):
        if self.model is None:
            return
        root = os.path.join(sys.path[0], "mnist_test")
        if not os.path.isdir(root):
            QMessageBox.warning(self, "提示", "测试集目录不存在: mnist_test")
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = []
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dp, f))
        if not files:
            QMessageBox.warning(self, "提示", "测试集图片为空")
            return
        path = files[np.random.randint(len(files))]
        self.image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(self.preview.width(), self.preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        data, arr = preprocess_image(path)
        label, conf, top, classes, probs = predict_with_confidence(self.model, data)
        if conf is None:
            self.result.setText(f"结果: {label}")
        else:
            top_str = ", ".join([f"{k}:{v:.4f}" for k, v in top])
            self.result.setText(f"结果: {label}  置信度: {conf:.4f}  Top-3: {top_str}")
        if probs is not None:
            self.ax.clear()
            n = len(probs)
            top_idx = list(np.argsort(probs)[::-1][:3])
            labels = [''] * n
            for i in top_idx:
                labels[i] = str(classes[i])
            colors = ['lightgray'] * n
            base_colors = ['tab:blue', 'tab:orange', 'tab:green']
            for j, i in enumerate(top_idx):
                colors[i] = base_colors[j % len(base_colors)]
            wedges, texts, autotexts = self.ax.pie(probs, labels=labels, colors=colors, autopct='%1.2f', startangle=90)
            for i in range(n):
                if i not in top_idx:
                    if i < len(texts):
                        texts[i].set_visible(False)
                    if i < len(autotexts):
                        autotexts[i].set_visible(False)
            self.ax.axis('equal')
            self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    w = PredictWidget()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
