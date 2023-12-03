# var
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import multiprocessing

# 仮想的なトレーニングデータ
X_train = np.random.rand(100, 20)
y_train = np.random.randint(2, size=100)

# 一つのテストデータ
x_test = np.random.rand(1, 20)

# 決定木モデルのリーフノード数を30に設定
base_model = DecisionTreeClassifier(max_leaf_nodes=30)

# 並列処理用のバギングクラシファイア
parallel_model = BaggingClassifier(base_model, n_estimators=100, n_jobs=multiprocessing.cpu_count())

# モデルの訓練
parallel_model.fit(X_train, y_train)

# 一つの入力に対して並列で推論
predictions_per_tree = [tree.predict(x_test) for tree in parallel_model.estimators_]

print("Predictions per Tree:")
for i, tree_predictions in enumerate(predictions_per_tree):
    print(f"Tree {i + 1}: {tree_predictions[0]}")

# それぞれの決定木が分類したクラスをリストに保存
classes_per_tree = [tree_predictions[0] for tree_predictions in predictions_per_tree]
print("Classes per Tree:", classes_per_tree)
