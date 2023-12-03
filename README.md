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


import numpy as np
from sklearn.decomposition import PCA

# 仮のデータを生成 (60000データポイント x 20特徴量)
data = np.random.rand(60000, 20)

# PCAの適用
pca = PCA()
pca.fit(data)

# 固有値と固有ベクトルを取得
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# 固有値を大きい順にソート
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[sorted_indices]

# 結果の表示
print("Sorted Eigenvalues:")
print(sorted_eigenvalues)

print("\nCorresponding Eigenvectors:")
print(sorted_eigenvectors)




from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 仮のデータを生成 (ここでは2クラス分類を想定)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルの構築
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルの訓練
rf_model.fit(X_train, y_train)

# テストデータに対する予測と確率の取得
predictions = rf_model.predict(X_test)
probabilities = rf_model.predict_proba(X_test)

# 結果の表示
print("Predicted Classes:")
print(predictions)

print("\nClass Probabilities:")
for i, class_probs in enumerate(probabilities):
    print(f"Sample {i + 1}: Class {rf_model.classes_[0]}: {class_probs[0]}, Class {rf_model.classes_[1]}: {class_probs[1]}")



from sklearn.naive_bayes import GaussianNB
import numpy as np

# ダミーデータとラベルの生成
X = np.random.rand(100, 10)  # 例として10個の特徴を持つ100個のデータ
y = np.random.randint(30, size=100)  # 30種類のクラス

# ナイーブベイズモデルの構築
naive_bayes_model = GaussianNB()

# モデルの訓練
naive_bayes_model.fit(X, y)

# ユーザーからの入力データ
user_input = np.random.rand(1, 10)  # 10個の特徴を持つデータとして仮定

# 分類と確率の取得
predicted_class = naive_bayes_model.predict(user_input)[0]
class_probabilities = naive_bayes_model.predict_proba(user_input)[0]

# 結果の表示
print("Predicted Class:", predicted_class)
print("Class Probabilities:")
for i, prob in enumerate(class_probabilities):
    print(f"Class {i}: {prob}")


