import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                           roc_curve, auc, precision_recall_curve,
                           log_loss, accuracy_score, f1_score,
                           precision_score, recall_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings
import re
from typing import List, Set

# --- Keras/TensorFlowのインポート ---
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import os

warnings.filterwarnings('ignore')

# 再現性のためのseed固定
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED) # TensorFlowのseedも固定

class JavaCodeTokenizer:
    """
    Javaのメソッド名やクラス名を適切にトークン化するカスタムトークナイザー

    主な機能:
    1. パッケージ名、クラス名、メソッド名の分離
    2. CamelCaseの分割
    3. アンダースコア区切りの分割
    4. 特殊文字の処理
    5. ストップワードの除去
    """

    def __init__(self, min_token_length: int = 2, include_package_tokens: bool = True):
        """
        初期化

        Parameters:
            min_token_length (int): 最小トークン長（これより短いトークンは除外）
            include_package_tokens (bool): パッケージ名のトークンを含めるかどうかのフラグ
        """
        self.min_token_length = min_token_length
        self.include_package_tokens = include_package_tokens
        # 汎用的なJavaのストップワードリスト（適宜追加・調整してください）
        self.java_stopwords = self._get_java_stopwords()

    def _get_java_stopwords(self) -> Set[str]:
        """Javaコードに特有のストップワードを定義"""
        # 一般的なキーワード、型、アクセス修飾子、予約語など
        return {
            "public", "private", "protected", "class", "interface", "enum",
            "void", "int", "long", "double", "float", "boolean", "char", "byte", "short",
            "String", "Object", "System", "out", "println", "new", "this", "super",
            "return", "if", "else", "for", "while", "do", "switch", "case", "default",
            "break", "continue", "try", "catch", "finally", "throw", "throws",
            "import", "package", "static", "final", "abstract", "transient", "volatile",
            "synchronized", "native", "strictfp", "extends", "implements", "instanceof",
            "assert", "const", "goto", "null", "true", "false",
            "get", "set", # getter/setterによく使われる
            "to", "of", "in", "on", "with", "from", "and", "or", "not", "is",
            "var", "let", "const", # 新しいJavaバージョンでのキーワード
            "util", "io", "lang", "math", "net", "sql", "awt", "swing", # 標準ライブラリの一部
            "value", "data", "info", "mgr", "impl", "factory", "builder", "adapter",
            "handler", "controller", "service", "dao", "repository", "entity", "dto",
            "model", "view", "component", "config", "init", "destroy", "main",
            "list", "map", "set", "array", "collection", "stream", "buffer",
            "reader", "writer", "input", "output", "exception", "error", "event",
            "create", "update", "delete", "remove", "add", "find", "load", "save",
            "run", "execute", "perform", "process", "handle", "check", "validate",
            "parse", "format", "convert", "build", "generate", "initiate", "start", "stop"
        }

    def tokenize(self, code_element_name: str) -> List[str]:
        """
        Javaのコード要素名（メソッド名、クラス名など）をトークン化する。
        """
        if not isinstance(code_element_name, str):
            return []

        tokens = []

        # まず '.' で分割し、パッケージ名と実際の名前を分離
        parts = code_element_name.split('.')
        if self.include_package_tokens:
            # パッケージ名もトークンとして含める場合
            for part in parts[:-1]: # 最後の要素（クラス/メソッド名）以外はパッケージ名
                tokens.extend(self._split_camel_case_and_underscore(part))

        # 最後の要素（クラス名またはメソッド名）を処理
        final_name = parts[-1]

        # カンマ以降（例: "method(arg1,arg2)"）を削除して名前のみにする
        final_name = final_name.split('(')[0]

        # 記号を除去し、CamelCaseとアンダースコアで分割
        cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '', final_name)
        split_tokens = self._split_camel_case_and_underscore(cleaned_name)
        tokens.extend(split_tokens)

        # ストップワード除去と最小トークン長フィルタリング
        filtered_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token not in self.java_stopwords and len(lower_token) >= self.min_token_length:
                filtered_tokens.append(lower_token)

        return filtered_tokens

    def _split_camel_case_and_underscore(self, name: str) -> List[str]:
        """CamelCaseとアンダースコアで文字列を分割するヘルパーメソッド"""
        # アンダースコアで分割
        words = name.split('_')
        split_words = []
        for word in words:
            # CamelCaseで分割
            camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)
            split_words.extend([s for s in camel_split if s]) # 空文字列を除外

        return split_words


class SimplifiedBugHunter:
    """
    カスタムJavaトークナイザー統合版ダウンサンプリング版バグ予測クラス
    （ニューラルネットワーク版）
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 1000,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 # NNのための新しいハイパーパラメータのデフォルト値
                 nn_epochs: int = 100,       # エポック数 (EarlyStoppingで調整)
                 nn_batch_size: int = 32):  # バッチサイズ

        self.best_model = None
        self.feature_importance = None # NNでは直接重要度を出すのは難しい
        self.all_feature_names = None
        self.selected_features = None # NN版では通常、全特徴量を選択する
        self.optimization_history = []
        self.best_params = None
        self.best_loss = float('inf')
        self.feature_selection_threshold = feature_selection_threshold
        self.initial_X = None
        self.initial_y = None
        self.tfidf_vectorizer_longname = None
        self.tfidf_vectorizer_parent = None
        self.tfidf_max_features = tfidf_max_features
        self.project_dummies_columns = None
        self.scaler = None

        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        self.downsampled_X = None
        self.downsampled_y = None
        self.original_class_distribution = None
        self.downsampled_train_distribution = None

        # NNのデフォルトハイパーパラメータを保存
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size

    def read_data(self, data_path: str) -> pd.DataFrame:
        """データを読み込む"""
        print(f"データファイル '{data_path}' を読み込み中...")
        data = pd.read_csv(data_path)
        print(f"データ読み込み完了。{len(data)}行を検出しました。")
        print(f"利用可能な列: {list(data.columns)}")
        return data

    def prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """
        データを前処理する。
        数値特徴量、TF-IDF特徴量、One-Hot Encoding特徴量を生成し、結合する。
        訓練時のみスケーラーやTF-IDFベクトライザーを学習する。
        """
        print("データ準備中...")

        # --- 修正箇所: ターゲット変数を 'Number of Bugs' に変更し、二値化 ---
        # 'Number of Bugs' をターゲット変数 y に設定し、0は0、1以上は1に変換
        y = (data['Number of Bugs'] > 0).astype(int).copy()
        # 'Number of Bugs' カラムを特徴量 X から削除
        X = data.drop(columns=['Number of Bugs']).copy()

        # 初期データを保存（トークナイザー分析用）
        if is_training:
            self.initial_X = X.copy()
            self.initial_y = y.copy()
        # ------------------------------------------------------------------

        # CSVファイルに実際に存在する数値特徴量を動的に特定
        # 文字列系のカラムを除外して、数値系のカラムのみを選択
        exclude_columns = ['Project', 'Hash', 'LongName', 'Parent']  # 文字列カラム
        numerical_feature_columns = []

        for col in X.columns:
            if col not in exclude_columns:
                # 数値型に変換可能かチェック
                try:
                    pd.to_numeric(X[col], errors='raise')
                    numerical_feature_columns.append(col)
                except (ValueError, TypeError):
                    # 数値に変換できない場合は除外
                    print(f"警告: {col} は数値特徴量として使用できません。スキップします。")
                    continue

        print(f"使用する数値特徴量: {numerical_feature_columns}")
        print(f"数値特徴量の数: {len(numerical_feature_columns)}")

        # 数値特徴量の処理
        if numerical_feature_columns:
            X_numerical = X[numerical_feature_columns].copy()
            # 数値型に変換
            for col in numerical_feature_columns:
                X_numerical[col] = pd.to_numeric(X_numerical[col], errors='coerce')

            X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
            X_numerical = X_numerical.fillna(0) # NaNを0で埋める

            if is_training:
                self.scaler = StandardScaler()
                X_numerical_scaled = self.scaler.fit_transform(X_numerical)
            else:
                if self.scaler is None:
                    raise ValueError("Scalerが学習されていません。訓練フェーズを先に実行してください。")
                X_numerical_scaled = self.scaler.transform(X_numerical)

            X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X.index)
        else:
            # 数値特徴量がない場合は空のDataFrameを作成
            X_numerical_scaled_df = pd.DataFrame(index=X.index)

        # TF-IDF特徴量の処理（LongName）
        if 'LongName' in X.columns:
            longname_data = X['LongName'].fillna("").astype(str)
            longname_tokenized = longname_data.apply(self.java_tokenizer.tokenize).apply(lambda x: " ".join(x))

            if is_training:
                self.tfidf_vectorizer_longname = TfidfVectorizer(max_features=self.tfidf_max_features)
                X_longname_tfidf = self.tfidf_vectorizer_longname.fit_transform(longname_tokenized)
            else:
                if self.tfidf_vectorizer_longname is None:
                    raise ValueError("LongName TF-IDF Vectorizerが学習されていません。訓練フェーズを先に実行してください。")
                X_longname_tfidf = self.tfidf_vectorizer_longname.transform(longname_tokenized)

            X_longname_tfidf_df = pd.DataFrame(X_longname_tfidf.toarray(),
                                               columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf.shape[1])],
                                               index=X.index)
        else:
            X_longname_tfidf_df = pd.DataFrame(index=X.index)
            print("警告: 'LongName'カラムが見つかりません。TF-IDF特徴量をスキップします。")

        # TF-IDF特徴量の処理（Parent）
        if 'Parent' in X.columns:
            parent_data = X['Parent'].fillna("").astype(str)
            parent_tokenized = parent_data.apply(self.java_tokenizer.tokenize).apply(lambda x: " ".join(x))

            if is_training:
                self.tfidf_vectorizer_parent = TfidfVectorizer(max_features=self.tfidf_max_features // 2) # Parentは少し少なく設定
                X_parent_tfidf = self.tfidf_vectorizer_parent.fit_transform(parent_tokenized)
            else:
                if self.tfidf_vectorizer_parent is None:
                    raise ValueError("Parent TF-IDF Vectorizerが学習されていません。訓練フェーズを先に実行してください。")
                X_parent_tfidf = self.tfidf_vectorizer_parent.transform(parent_tokenized)

            X_parent_tfidf_df = pd.DataFrame(X_parent_tfidf.toarray(),
                                             columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf.shape[1])],
                                             index=X.index)
        else:
            X_parent_tfidf_df = pd.DataFrame(index=X.index)
            print("警告: 'Parent'カラムが見つかりません。Parent TF-IDF特徴量をスキップします。")

        # One-Hot Encoding（Project）
        if 'Project' in X.columns:
            project_data = X['Project'].fillna("Unknown").astype(str)
            if is_training:
                X_project_onehot = pd.get_dummies(project_data, prefix='Project', dtype=int)
                # 訓練データからダミー変数のカラム名を保存
                self.project_dummies_columns = X_project_onehot.columns
            else:
                X_project_onehot = pd.get_dummies(project_data, prefix='Project', dtype=int)
                # 訓練時に保存したカラムに再インデックス化し、存在しないプロジェクトは0で埋める
                X_project_onehot = X_project_onehot.reindex(columns=self.project_dummies_columns, fill_value=0)
        else:
            X_project_onehot = pd.DataFrame(index=X.index)
            print("警告: 'Project'カラムが見つかりません。One-Hot Encoding特徴量をスキップします。")

        # 全特徴量を結合
        X_processed = pd.concat([X_numerical_scaled_df, X_longname_tfidf_df, X_parent_tfidf_df, X_project_onehot], axis=1)

        self.all_feature_names = X_processed.columns.tolist()

        print(f"データ準備完了。特徴量数: {len(self.all_feature_names)}")
        print(f"  - 数値特徴量: {len(numerical_feature_columns)}")
        print(f"  - LongName TF-IDF: {X_longname_tfidf_df.shape[1] if not X_longname_tfidf_df.empty else 0}")
        print(f"  - Parent TF-IDF: {X_parent_tfidf_df.shape[1] if not X_parent_tfidf_df.empty else 0}")
        print(f"  - Project One-Hot: {X_project_onehot.shape[1] if not X_project_onehot.empty else 0}")

        return X_processed, y

    def apply_downsampling(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        少数派クラスのサンプル数を多数派クラスに合わせるためにダウンサンプリングを適用する。
        """
        print("ダウンサンプリング処理中...")

        # 元のクラス分布を保存
        class_0_original = y[y == 0].count()
        class_1_original = y[y == 1].count()
        self.original_class_distribution = {
            'class_0': class_0_original,
            'class_1': class_1_original,
            'total': len(y)
        }

        print(f"元のクラス分布: クラス0={class_0_original}, クラス1={class_1_original}")

        # クラス0とクラス1のインデックスを分離
        X_class_0 = X[y == 0]
        y_class_0 = y[y == 0]
        X_class_1 = X[y == 1]
        y_class_1 = y[y == 1]

        # 多数派クラス (通常はクラス0) のサンプル数を少数派クラス (通常はクラス1) に合わせる
        # ここでは、常にクラス1の数にクラス0をダウンサンプリング
        n_samples_class_1 = len(X_class_1)

        if n_samples_class_1 == 0:
            print("警告: 少数派クラス (バグあり) のサンプルがありません。ダウンサンプリングをスキップします。")
            self.downsampled_train_distribution = self.original_class_distribution
            return X, y

        if len(X_class_0) > n_samples_class_1:
            X_class_0_downsampled, y_class_0_downsampled = resample(
                X_class_0, y_class_0,
                replace=False,    # 非復元抽出
                n_samples=n_samples_class_1, # 少数派クラスの数に合わせる
                random_state=GLOBAL_SEED # 再現性のためのシード固定
            )
            print(f"クラス0を {len(X_class_0)} 件から {len(X_class_0_downsampled)} 件にダウンサンプリングしました。")
        else:
            X_class_0_downsampled = X_class_0
            y_class_0_downsampled = y_class_0
            print("クラス0のサンプル数が少数派クラス以下なので、ダウンサンプリングは不要です。")

        # ダウンサンプリングされた多数派クラスと少数派クラスを結合
        X_downsampled = pd.concat([X_class_0_downsampled, X_class_1])
        y_downsampled = pd.concat([y_class_0_downsampled, y_class_1])

        # シャッフル (重要: クラスの偏りをなくすため)
        shuffled_indices = np.random.permutation(len(X_downsampled))
        self.downsampled_X = X_downsampled.iloc[shuffled_indices].reset_index(drop=True)
        self.downsampled_y = y_downsampled.iloc[shuffled_indices].reset_index(drop=True)

        # ダウンサンプリング後の訓練データ分布を保存
        self.downsampled_train_distribution = {
            'class_0': self.downsampled_y[self.downsampled_y == 0].count(),
            'class_1': self.downsampled_y[self.downsampled_y == 1].count(),
            'total': len(self.downsampled_y)
        }

        print(f"ダウンサンプリング完了。新しいデータセットサイズ: {len(self.downsampled_y)}行")
        print(f"新しいクラス分布: クラス0={self.downsampled_train_distribution['class_0']}、クラス1={self.downsampled_train_distribution['class_1']}")

        return self.downsampled_X, self.downsampled_y

    def _build_nn_model(self, input_dim: int, params: dict) -> keras.Model:
        """ニューラルネットワークモデルの構築"""
        model = Sequential()

        # 隠れ層の数 (最小1層)
        num_hidden_layers = int(params.get('nn_hidden_layers', 1))
        # 各隠れ層のニューロン数
        neurons = int(params.get('nn_neurons', 64))
        # ドロップアウト率
        dropout_rate = params.get('dropout_rate', 0.2) # デフォルト値を設定

        # 最初の隠れ層
        model.add(Dense(neurons, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(dropout_rate))

        # 残りの隠れ層
        for _ in range(num_hidden_layers - 1):
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(dropout_rate))

        # 出力層 (二値分類なのでsigmoid)
        model.add(Dense(1, activation='sigmoid'))

        # モデルのコンパイル
        optimizer = Adam(learning_rate=params.get('nn_learning_rate', 0.001))
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', # 二値分類の標準的な損失関数
                      metrics=['accuracy'])
        return model

    def evaluate_model_with_cv(self, params: dict, X: pd.DataFrame, y: pd.Series,
                              k_folds: int = 3) -> float:
        """交差検証を用いたモデル評価（ニューラルネットワーク版）"""
        try:
            # XをNumPy配列に変換 (NNの入力形式に合わせる)
            X_np = X.to_numpy()
            y_np = y.to_numpy()

            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
            total_loss = 0.0

            # EarlyStoppingコールバックの設定 (過学習防止)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            for train_idx, val_idx in cv.split(X_np, y_np):
                X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
                y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]

                # モデル構築
                model = self._build_nn_model(X_train_fold.shape[1], params)

                # モデル学習
                history = model.fit(X_train_fold, y_train_fold,
                                    epochs=self.nn_epochs, # nn_epochsは__init__で設定したデフォルト値を使用
                                    batch_size=self.nn_batch_size, # nn_batch_sizeは__init__で設定したデフォルト値を使用
                                    validation_data=(X_val_fold, y_val_fold),
                                    callbacks=[early_stopping], # EarlyStoppingを適用
                                    verbose=0) # 学習中の出力を抑制

                # 最良エポックのval_lossを取得
                fold_loss = min(history.history['val_loss'])
                total_loss += fold_loss

            avg_loss = total_loss / k_folds
            return avg_loss

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')


    def optimize_hyperparameters_with_log_loss(self, X: pd.DataFrame, y: pd.Series,
                                                max_iterations: int = 30) -> dict:
        """Log Loss損失関数を用いたベイジアン最適化（ニューラルネットワーク版）"""
        print("\n=== Log Lossベース ベイジアン最適化（ニューラルネットワーク使用）===")
        print("最適化手法: Bayesian Optimization (scikit-optimize)")
        print("探索パラメータ: nn_hidden_layers, nn_neurons, nn_learning_rate, dropout_rate")
        print("クラス不均衡対応: ダウンサンプリング (事前に適用済み)")
        print("特徴量: 数値 + Java TF-IDF + One-Hot Encoding + 正規化")

        self.best_loss = float('inf')
        self.best_params = None
        self.optimization_history = []

        # 探索空間の定義 (NN用)
        search_space = [
            Integer(1, 3, name='nn_hidden_layers'), # 隠れ層の数: 1から3層
            Integer(32, 256, name='nn_neurons'),    # 各隠れ層のニューロン数: 32から256
            Real(1e-4, 1e-2, prior='log-uniform', name='nn_learning_rate'), # 学習率: 0.0001から0.01 (対数均等に)
            Real(0.1, 0.5, name='dropout_rate') # ドロップアウト率: 0.1から0.5
        ]

        # 目的関数
        def objective(params):
            nn_hidden_layers, nn_neurons, nn_learning_rate, dropout_rate = params

            param_dict = {
                'nn_hidden_layers': int(nn_hidden_layers),
                'nn_neurons': int(nn_neurons),
                'nn_learning_rate': float(nn_learning_rate),
                'dropout_rate': float(dropout_rate),
            }

            # モデル評価
            loss = self.evaluate_model_with_cv(param_dict, X, y)

            # 履歴記録
            self.optimization_history.append({
                'params': param_dict.copy(),
                'loss': loss
            })

            # 最良パラメータ更新
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = param_dict.copy()
                print(f"新しい最良損失: {loss:.4f}")
                print(f"パラメータ: {param_dict}")

            return loss

        # Bayesian Optimization実行
        print("Bayesian Optimization開始...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=max_iterations,
            random_state=GLOBAL_SEED,
            acq_func='EI',
            n_initial_points=5
        )

        print(f"\nBayesian Optimization完了!")
        print(f"最良損失: {self.best_loss:.4f}")
        final_best_params = self.best_params.copy() if self.best_params else {}
        print(f"最良パラメータ: {final_best_params}")
        print(f"総評価回数: {len(self.optimization_history)}")

        return final_best_params

    def train_initial_model_for_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        ニューラルネットワークでは特徴量重要度を直接計算することが難しい。
        ここでは、特徴量選択のために初期モデルを学習するステップをスキップする。
        """
        print("\n=== ニューラルネットワークでは特徴量重要度を直接計算しないため、このステップはスキップします。===")
        self.feature_importance = None # 明示的にNoneに設定
        # selected_featuresを全特徴量に設定
        self.selected_features = X.columns.tolist()
        print("特徴量選択は全特徴量を使用します。")


    def select_features_by_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ニューラルネットワークでは、このステップはそのままでは適用できない。
        今回は簡略化のため、全特徴量を選択することにする。
        """
        print(f"\n=== ニューラルネットワークでは特徴量削減をスキップし、全特徴量を使用します。 ===")

        # train_initial_model_for_feature_importanceで設定されているはず
        if self.selected_features is None:
            self.selected_features = X.columns.tolist()

        X_selected = X[self.selected_features]

        print(f"選択された特徴量数: {len(self.selected_features)}")

        # 特徴量タイプ別の統計 (これはそのまま使える)
        longname_count = len([f for f in self.selected_features if f.startswith('LongName_tfidf_')])
        parent_count = len([f for f in self.selected_features if f.startswith('Parent_tfidf_')])
        project_count = len([f for f in self.selected_features if f.startswith('Project_')])
        numerical_count = len(self.selected_features) - longname_count - parent_count - project_count

        print(f"  - 数値特徴量: {numerical_count}")
        print(f"  - LongName TF-IDF: {longname_count}")
        print(f"  - Parent TF-IDF: {parent_count}")
        print(f"  - Project One-Hot: {project_count}")

        return X_selected


    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, optimal_params: dict):
        """最適化されたパラメータでニューラルネットワークモデルを学習"""
        print("\n=== 最適化ニューラルネットワークモデル学習（カスタムJavaトークナイザー版）===")

        # X, y をNumPy配列に変換
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # モデル構築
        model = self._build_nn_model(X_np.shape[1], optimal_params)

        # EarlyStoppingコールバックの設定 (val_lossが5エポック改善しなかったら停止)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # モデル学習
        print(f"学習パラメータ: Epochs={self.nn_epochs}, Batch Size={self.nn_batch_size}")
        print(f"NN構造: 隠れ層={optimal_params.get('nn_hidden_layers')}, ニューロン数={optimal_params.get('nn_neurons')}")
        print(f"学習率={optimal_params.get('nn_learning_rate'):.6f}, ドロップアウト率={optimal_params.get('dropout_rate'):.2f}")


        history = model.fit(X_np, y_np,
                            epochs=self.nn_epochs,
                            batch_size=self.nn_batch_size,
                            validation_split=0.2, # 訓練データの一部をバリデーションに使用
                            callbacks=[early_stopping],
                            verbose=1) # 学習中の出力を表示

        self.best_model = model

        print("最適化モデル学習完了")
        # ニューラルネットワークではfeature_importanceを直接取得できないためNoneのまま
        self.feature_importance = None
        return self.best_model

    def comprehensive_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """包括的評価（デフォルトしきい値0.5のみ使用）"""
        print("\n=== 包括的評価 ===")

        # X_testをNumPy配列に変換
        X_test_np = X_test.to_numpy()

        # 予測確率取得
        y_pred_proba = self.best_model.predict(X_test_np).flatten()

        # デフォルトしきい値（0.5）での予測
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 評価指標の計算
        results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'LogLoss': log_loss(y_test, y_pred_proba), # Kerasのlog_lossではなくsklearnのlog_lossを使用
            'Threshold': 0.5
        }

        print("\n--- 評価結果（しきい値=0.5）---")
        for metric, score in results.items():
            if metric != 'Threshold':
                print(f"{metric}: {score:.4f}")

        return results, y_pred, y_pred_proba

    def plot_results(self, results: dict, y_test: pd.Series, y_pred: np.ndarray,
                    y_pred_proba: np.ndarray, X_test: pd.DataFrame):
        """結果の可視化（ニューラルネットワーク版、特徴量重要度表示は調整）"""
        print("\n=== 結果可視化 ===")

        plt.figure(figsize=(18, 15)) # 全体的なサイズを調整

        # 1. 混同行列
        plt.subplot(3, 3, 1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                              display_labels=[0, 1])
        display_matrix.plot(ax=plt.gca(), cmap='Blues')
        plt.title(f"混同行列（しきい値=0.5）")

        # 2. ROC曲線
        plt.subplot(3, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC曲線 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('偽陽性率')
        plt.ylabel('真陽性率')
        plt.title('ROC曲線')
        plt.legend(loc="lower right")

        # 3. Precision-Recall曲線
        plt.subplot(3, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall曲線')
        plt.grid(True, alpha=0.3)

        # 4. 評価指標表示
        plt.subplot(3, 3, 4)
        metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'LogLoss']
        values = [results.get(metric, 0) for metric in metrics]

        bars = plt.bar(metrics, values, color='lightblue', alpha=0.8)
        plt.title('評価指標')
        plt.ylabel('スコア')
        plt.ylim(0, 1.0) # LogLossは1以上になることもあるので注意が必要だが、一般的には小さい方が良い
        # LogLossの値が極端に大きい場合を考慮してy軸の範囲を調整
        max_logloss = max(values) if 'LogLoss' in metrics else 0.0
        plt.ylim(0, max(1.0, max_logloss * 1.1)) # 最小0、最大で一番大きい値の1.1倍

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}' if value <=1.0 else f'{value:.2f}', ha='center', va='bottom', fontsize=10)


        # 5. 特徴量重要度（NNでは直接的な重要度がないため、表示を調整）
        plt.subplot(3, 3, 5)
        plt.text(0.5, 0.5, 'ニューラルネットワークでは\n直接的な特徴量重要度は\n計算されません',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12, color='gray')
        plt.title('特徴量重要度 (NN)')
        plt.axis('off')


        # 6. ベイジアン最適化履歴
        plt.subplot(3, 3, 6)
        if self.optimization_history:
            losses = [entry['loss'] for entry in self.optimization_history]
            iterations = range(len(losses))

            plt.plot(iterations, losses, 'bo-', linewidth=2, markersize=4,
                    alpha=0.7, label='評価点')

            # 最良値の推移
            cumulative_best = []
            current_best = float('inf')
            for loss in losses:
                if loss < current_best:
                    current_best = loss
                cumulative_best.append(current_best)

            plt.plot(iterations, cumulative_best, 'r-', linewidth=3,
                    label='最良値推移')

            plt.xlabel('評価回数')
            plt.ylabel('Log Loss値')
            plt.title('ベイジアン最適化過程')
            plt.grid(True, alpha=0.3)
            plt.legend()

        # 7. ダウンサンプリング効果の可視化
        plt.subplot(3, 3, 7)
        if self.original_class_distribution and self.downsampled_train_distribution:
            original_counts_total = [self.original_class_distribution['class_0'],
                                     self.original_class_distribution['class_1']]
            downsampled_counts_train = [self.downsampled_train_distribution['class_0'],
                                        self.downsampled_train_distribution['class_1']]

            x = np.arange(2)
            width = 0.35

            plt.bar(x - width/2, original_counts_total, width, label='元データ全体', alpha=0.7, color='lightcoral')
            plt.bar(x + width/2, downsampled_counts_train, width, label='ダウンサンプリング後訓練データ', alpha=0.7, color='lightblue')

            plt.xlabel('クラス')
            plt.ylabel('サンプル数')
            plt.title('ダウンサンプリング効果')
            plt.xticks(x, ['クラス 0', 'クラス 1'])
            plt.legend()

            for i, (orig, down) in enumerate(zip(original_counts_total, downsampled_counts_train)):
                plt.text(i - width/2, orig + max(original_counts_total) * 0.01, f'{orig}',
                        ha='center', va='bottom', fontsize=9)
                plt.text(i + width/2, down + max(original_counts_total) * 0.01, f'{down}',
                        ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, 'ダウンサンプリング情報なし',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=12, color='gray')
            plt.title('ダウンサンプリング効果')
            plt.axis('off')

        # 8. 予測確率分布
        plt.subplot(3, 3, 8)
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='実際のクラス 0', color='blue')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='実際のクラス 1', color='red')
        plt.axvline(x=0.5, color='gray', linestyle='--', label='しきい値 0.5')
        plt.xlabel('予測確率')
        plt.ylabel('頻度')
        plt.title('予測確率分布')
        plt.legend()

        # 9. サマリー表示
        plt.subplot(3, 3, 9)
        plt.axis('off')

        feature_count = len(self.selected_features) if self.selected_features else 0

        if self.original_class_distribution and self.downsampled_train_distribution:
            reduction_rate = (1 - self.downsampled_train_distribution['total'] / self.original_class_distribution['total']) * 100
            downsample_info = f"ダウンサンプリング: {reduction_rate:.1f}%削減"
        else:
            downsample_info = "ダウンサンプリング: 情報なし"

        if self.selected_features:
            longname_count = len([f for f in self.selected_features if f.startswith('LongName_tfidf_')])
            parent_count = len([f for f in self.selected_features if f.startswith('Parent_tfidf_')])
            project_count = len([f for f in self.selected_features if f.startswith('Project_')])
            numerical_count = feature_count - longname_count - parent_count - project_count
        else:
            longname_count = parent_count = project_count = numerical_count = 0

        # NNのハイパーパラメータを表示に追加
        nn_params = results.get('optimal_params', {})
        summary_text = f"""
ニューラルネットワーク版バグ予測結果

使用特徴量数: {feature_count}個
  数値: {numerical_count}, LongName TF-IDF: {longname_count}
  Parent TF-IDF: {parent_count}, Project: {project_count}
TF-IDF最大特徴量数: {self.tfidf_max_features}
{downsample_info}
使用しきい値: 0.500 (固定)

最適化NNパラメータ:
  隠れ層: {self.best_params.get('nn_hidden_layers', 'N/A')}
  ニューロン: {self.best_params.get('nn_neurons', 'N/A')}
  学習率: {self.best_params.get('nn_learning_rate', 'N/A'):.6f}
  ドロップアウト率: {self.best_params.get('dropout_rate', 'N/A'):.2f}
  バッチサイズ: {self.nn_batch_size} (固定)
  最大エポック: {self.nn_epochs} (EarlyStopping適用)

評価結果（テストデータ）:
F1スコア: {results['F1']:.3f}
Precision: {results['Precision']:.3f}
Recall: {results['Recall']:.3f}
Accuracy: {results['Accuracy']:.3f}
Log Loss: {results['LogLoss']:.3f}

ニューラルネットワークでJavaコードの
特徴を学習し、バグを予測
        """
        plt.text(0.1, 0.5, summary_text, fontsize=8,
                verticalalignment='center', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

    def run_pipeline(self, data_path: str):
        """ニューラルネットワークを組み込んだパイプラインの実行"""
        print("=== ニューラルネットワーク版ダウンサンプリングバグ予測パイプライン ===")

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備（カスタムJavaトークナイザー、TF-IDF特徴量、One-Hot Encoding特徴量、正規化を使用）
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. ダウンサンプリングをここで一度だけ適用
        X_downsampled, y_downsampled = self.apply_downsampling(X_full, y_full)
        print(f"ダウンサンプリング後データセットサイズ: {len(X_downsampled)}行")

        # 4. データ分割（ダウンサンプリング後のデータに対して）
        X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(
            X_downsampled, y_downsampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_downsampled
        )
        print(f"訓練データ (ダウンサンプリング後): {len(X_train_ds)}行, テストデータ (ダウンサンプリング後): {len(X_test_ds)}行")

        # 5. 特徴量重要度取得のための初期モデル学習（NNではこのステップをスキップ）
        self.train_initial_model_for_feature_importance(X_train_ds, y_train_ds)


        # 6. 特徴量削減（NN版では全特徴量を使用するように変更）
        # self.selected_featuresがこの時点でセットされているはず
        X_train_reduced = self.select_features_by_importance(X_train_ds)
        X_test_reduced = self.select_features_by_importance(X_test_ds)


        # 7. Log Lossベースハイパーパラメータ最適化（削減された訓練データで実施）
        optimal_params = self.optimize_hyperparameters_with_log_loss(
            X_train_reduced, y_train_ds, max_iterations=30
        )

        # 最適化されたパラメータをクラスインスタンスに保存
        self.best_params = optimal_params

        # 8. 最適化モデル学習（削減された訓練データで実施）
        optimized_model = self.train_optimized_model(
            X_train_reduced, y_train_ds, optimal_params
        )

        # 9. 評価（削減されたテストデータで実施）
        results, y_pred, y_pred_proba = self.comprehensive_evaluation(
            X_test_reduced, y_test_ds
        )
        # 結果に最適なパラメータを追加 (プロットのサマリー表示用)
        results['optimal_params'] = optimal_params

        # 10. 結果可視化
        self.plot_results(
            results, y_test_ds, y_pred, y_pred_proba, X_test_reduced
        )

        return results, optimal_params

    def predict(self, X: pd.DataFrame) -> tuple:
        """予測（ニューラルネットワーク版）"""
        if self.best_model is None:
            raise ValueError("モデルが学習されていません。まずrun_pipeline()を実行してください。")
        if self.selected_features is None:
            raise ValueError("特徴量削減が実行されていません。まずrun_pipeline()を実行してください。")
        if self.tfidf_vectorizer_longname is None or self.tfidf_vectorizer_parent is None:
            raise ValueError("TF-IDF Vectorizerが学習されていません。まずrun_pipeline()を実行してください。")
        if self.project_dummies_columns is None:
            raise ValueError("One-Hot Encoderが学習されていません。まずrun_pipeline()を実行してください。")
        if self.scaler is None:
            raise ValueError("Scalerが学習されていません。まずrun_pipeline()を実行してください。")

        # データ準備（推論時はis_training=False）
        X_processed_full, _ = self.prepare_data(X, is_training=False)

        # 学習時に選択された特徴量のみを選択 (NN版では全特徴量を選択する前提)
        # prepare_dataで既に全特徴量を処理済みなので、ここでは単に整合性を保つ
        if self.selected_features and len(self.selected_features) > 0:
            X_processed = X_processed_full[self.selected_features]
        else:
            # selected_featuresが空またはNoneの場合、全特徴量を使用
            X_processed = X_processed_full
            print("警告: selected_featuresが設定されていないため、全ての処理済み特徴量を使用します。")

        # NumPy配列に変換して予測
        y_pred_proba = self.best_model.predict(X_processed.to_numpy()).flatten()

        # デフォルトしきい値での予測
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred, y_pred_proba

    def get_feature_analysis(self) -> dict:
        """特徴量分析結果の取得（NN版に合わせて調整）"""
        params_to_return = self.best_params.copy() if self.best_params else {}

        downsampling_info = {}
        if self.original_class_distribution and self.downsampled_train_distribution:
            downsampling_info = {
                'original_class_0': self.original_class_distribution['class_0'],
                'original_class_1': self.original_class_distribution['class_1'],
                'original_total': self.original_class_distribution['total'],
                'downsampled_train_class_0': self.downsampled_train_distribution['class_0'],
                'downsampled_train_class_1': self.downsampled_train_distribution['class_1'],
                'downsampled_train_total': self.downsampled_train_distribution['total'],
                'reduction_rate': (1 - self.downsampled_train_distribution['total'] / self.original_class_distribution['total']) * 100
            }

        return {
            'best_params': params_to_return,
            'optimization_history': self.optimization_history,
            'feature_importance': None, # NNでは直接取得できない
            'selected_features': self.selected_features,
            'all_feature_names': self.all_feature_names,
            'feature_selection_threshold': self.feature_selection_threshold,
            'tfidf_max_features': self.tfidf_max_features,
            'downsampling_info': downsampling_info,
            'java_tokenizer_settings': {
                'min_token_length': self.java_tokenizer.min_token_length,
                'include_package_tokens': self.java_tokenizer.include_package_tokens,
                'stopwords_count': len(self.java_tokenizer.java_stopwords)
            }
        }

    def display_feature_importance_table(self, top_n: int = 10):
        """特徴量重要度テーブルの表示（NN版に合わせて調整）"""
        print("\n=== ニューラルネットワークでは直接的な特徴量重要度テーブルは表示できません。===")
        print("他の手法（例：SHAP値など）を用いて、特徴量の影響を分析できます。")

    def display_downsampling_summary(self):
        """ダウンサンプリング結果のサマリーを表示する"""
        print("\n=== ダウンサンプリングサマリー ===")
        if self.original_class_distribution and self.downsampled_train_distribution:
            orig_0 = self.original_class_distribution['class_0']
            orig_1 = self.original_class_distribution['class_1']
            orig_total = self.original_class_distribution['total']
            down_0 = self.downsampled_train_distribution['class_0']
            down_1 = self.downsampled_train_distribution['class_1']
            down_total = self.downsampled_train_distribution['total']

            print(f"元データ合計: {orig_total} (クラス0: {orig_0}, クラス1: {orig_1})")
            print(f"ダウンサンプリング後訓練データ合計: {down_total} (クラス0: {down_0}, クラス1: {down_1})")

            if orig_total > 0:
                reduction_rate = (1 - down_total / orig_total) * 100
                print(f"データ削減率: {reduction_rate:.1f}%")
            else:
                print("元のデータが空です。")
        else:
            print("ダウンサンプリング情報はまだ利用できません。")

    def display_tokenizer_analysis(self, sample_size: int = 3):
        """トークナイザーの動作例と分析を表示する"""
        print("\n=== Javaコードトークナイザー分析 ===")
        print(f"最小トークン長: {self.java_tokenizer.min_token_length}")
        print(f"パッケージ名トークンを含める: {self.java_tokenizer.include_package_tokens}")
        print(f"ストップワード数: {len(self.java_tokenizer.java_stopwords)}")

        # データが読み込まれているか確認
        if self.initial_X is None:
            print("データがまだ読み込まれていないため、トークナイザーの動作例を表示できません。")
            return

        print(f"\n--- サンプルメソッド名（{sample_size}件）とトークン化結果 ---")
        sample_methods = self.initial_X['LongName'].fillna("").astype(str).sample(min(sample_size, len(self.initial_X)), random_state=GLOBAL_SEED)

        for i, method_name in enumerate(sample_methods):
            tokens = self.java_tokenizer.tokenize(method_name)
            print(f"  [{i+1}]")
            print(f"    元メソッド名: {method_name}")
            print(f"    トークン化結果: {tokens}")
            print("-" * 30)

        print(f"\n--- サンプル親要素名（{sample_size}件）とトークン化結果 ---")
        sample_parents = self.initial_X['Parent'].fillna("").astype(str).sample(min(sample_size, len(self.initial_X)), random_state=GLOBAL_SEED)

        for i, parent_name in enumerate(sample_parents):
            tokens = self.java_tokenizer.tokenize(parent_name)
            print(f"  [{i+1}]")
            print(f"    元親要素名: {parent_name}")
            print(f"    トークン化結果: {tokens}")
            print("-" * 30)

# 使用例 (if __name__ == "__main__": ブロック)
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # カスタムJavaトークナイザー統合版バグハンターのインスタンス作成
    bug_hunter = SimplifiedBugHunter(
        feature_selection_threshold=0.001, # NN版ではあまり意味をなさないが、引数として残しておく
        tfidf_max_features=1000,
        java_tokenizer_min_length=2,
        include_package_tokens=False, # パッケージ名を除外して、より重要な部分に集中
        # ニューラルネットワークのハイパーパラメータの固定値（最適化対象外）
        nn_epochs=100, # EarlyStoppingで早期終了するので、大きめに設定
        nn_batch_size=64
    )

    # パイプライン実行
    results, optimal_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("ニューラルネットワーク版バグ予測完了!")
    print("="*60)
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"Log Loss: {results['LogLoss']:.3f}")


    # ダウンサンプリングサマリーの表示
    bug_hunter.display_downsampling_summary()

    # 特徴量重要度テーブルの表示 (NN版ではスキップされることを明示)
    bug_hunter.display_feature_importance_table(top_n=15)

    # トークナイザーの動作例表示
    bug_hunter.display_tokenizer_analysis(sample_size=3)

    feature_analysis = bug_hunter.get_feature_analysis()
    print(f"\n最適パラメータ (NN): {feature_analysis['best_params']}")
    print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}") # NN版では全特徴量
    print(f"全特徴量数 (Java TF-IDF含む): {len(feature_analysis['all_feature_names'])}")
    print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")
    print(f"Javaトークナイザー設定: {feature_analysis['java_tokenizer_settings']}")

    if feature_analysis['downsampling_info']:
        ds_info = feature_analysis['downsampling_info']
        print(f"データ削減率: {ds_info['reduction_rate']:.1f}%")

    print("\n" + "="*60)
