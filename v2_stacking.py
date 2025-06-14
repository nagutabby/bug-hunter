import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import re

# 新規追加のインポート
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from typing import Set, Dict, Any, Tuple

import warnings
warnings.filterwarnings('ignore')

# 再現性のためのseed固定
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


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
            include_package_tokens (bool): パッケージ名のトークンを含めるかどうか
        """
        self.min_token_length = min_token_length
        self.include_package_tokens = include_package_tokens

        # Javaでよく使われる一般的な単語（ストップワード）
        self.java_stopwords: Set[str] = {
            'java', 'util', 'lang', 'io', 'net', 'org', 'com', 'javax',
            'get', 'set', 'is', 'has', 'to', 'from', 'with', 'without',
            'init', 'new', 'create', 'build', 'make', 'do', 'run', 'execute',
            'class', 'interface', 'abstract', 'final', 'static', 'public',
            'private', 'protected', 'void', 'int', 'string', 'boolean',
            'double', 'float', 'long', 'short', 'byte', 'char',
            'impl', 'default', 'base', 'simple', 'empty'
        }

    def _split_camel_case(self, text: str) -> list[str]:
        """
        CamelCaseの文字列を単語に分割

        例: "GeoPointDoubleArray" -> ["Geo", "Point", "Double", "Array"]
        """
        # 大文字の前で分割（先頭は除く）
        parts = re.sub(r'(?<!^)(?=[A-Z])', ' ', text).split()
        return [part for part in parts if len(part) >= self.min_token_length]

    def _split_snake_case(self, text: str) -> list[str]:
        """
        snake_caseやkebab-caseの文字列を単語に分割

        例: "atomic_field_data" -> ["atomic", "field", "data"]
        """
        parts = re.split(r'[_\-]', text)
        return [part for part in parts if len(part) >= self.min_token_length]

    def _extract_method_signature_tokens(self, method_signature: str) -> list[str]:
        """
        メソッドシグネチャからトークンを抽出

        例: "org.elasticsearch.index.fielddata.plain.GeoPointDoubleArrayAtomicFieldData$Empty.<init>()V"
        """
        tokens = []

        # メソッドシグネチャのパターンマッチング
        # パッケージ.クラス名.メソッド名(引数)戻り値型 の形式
        pattern = r'^(.*?)\.([^.()]+)\(([^)]*)\)(.*)$'
        match = re.match(pattern, method_signature)

        if match:
            package_and_class = match.group(1)  # パッケージ+クラス部分
            method_name = match.group(2)        # メソッド名
            parameters = match.group(3)         # 引数部分
            return_type = match.group(4)        # 戻り値型

            # パッケージとクラス名の処理
            if package_and_class:
                package_parts = package_and_class.split('.')

                # パッケージ名の処理（オプション）
                if self.include_package_tokens:
                    for part in package_parts[:-1]:  # 最後はクラス名なので除外
                        if part and len(part) >= self.min_token_length:
                            tokens.append(part.lower())

                # クラス名の処理（最後の部分）
                if package_parts:
                    class_name = package_parts[-1]
                    # 内部クラス記号 $ の処理
                    class_parts = class_name.split('$')
                    for class_part in class_parts:
                        if class_part:
                            # CamelCaseの分割
                            camel_tokens = self._split_camel_case(class_part)
                            tokens.extend([token.lower() for token in camel_tokens])

            # メソッド名の処理
            if method_name and method_name != '<init>' and method_name != '<clinit>':
                # アンダースコアで分割
                snake_tokens = self._split_snake_case(method_name)
                for token in snake_tokens:
                    # CamelCaseの分割
                    camel_tokens = self._split_camel_case(token)
                    tokens.extend([token.lower() for token in camel_tokens])
            elif method_name in ['<init>', '<clinit>']:
                tokens.append('constructor')

        else:
            # パターンにマッチしない場合は単純に分割
            # ドット区切りで分割
            parts = method_signature.split('.')
            for part in parts:
                if part and '(' not in part:  # 引数部分を除外
                    # CamelCaseとsnake_caseの両方で分割
                    snake_tokens = self._split_snake_case(part)
                    for token in snake_tokens:
                        camel_tokens = self._split_camel_case(token)
                        tokens.extend([token.lower() for token in camel_tokens])

        return tokens

    def _extract_class_name_tokens(self, class_name: str) -> list[str]:
        """
        クラス名からトークンを抽出

        例: "org.elasticsearch.index.fielddata.plain.GeoPointDoubleArrayAtomicFieldData$Empty"
        """
        tokens = []

        # パッケージとクラス名を分離
        parts = class_name.split('.')

        # パッケージ名の処理（オプション）
        if self.include_package_tokens:
            for part in parts[:-1]:  # 最後はクラス名なので除外
                if part and len(part) >= self.min_token_length:
                    tokens.append(part.lower())

        # クラス名の処理（最後の部分）
        if parts:
            class_part = parts[-1]
            # 内部クラス記号 $ の処理
            class_components = class_part.split('$')
            for component in class_components:
                if component:
                    # CamelCaseの分割
                    camel_tokens = self._split_camel_case(component)
                    tokens.extend([token.lower() for token in camel_tokens])

        return tokens

    def tokenize_longname(self, longname: str) -> list[str]:
        """
        LongName列の値をトークン化（メソッドシグネチャ）
        """
        if not longname or pd.isna(longname):
            return []

        tokens = self._extract_method_signature_tokens(str(longname))

        # ストップワードの除去と最小長チェック
        filtered_tokens = [
            token for token in tokens
            if len(token) >= self.min_token_length and token.lower() not in self.java_stopwords
        ]

        return filtered_tokens

    def tokenize_parent(self, parent: str) -> list[str]:
        """
        Parent列の値をトークン化（クラス名）
        """
        if not parent or pd.isna(parent):
            return []

        tokens = self._extract_class_name_tokens(str(parent))

        # ストップワードの除去と最小長チェック
        filtered_tokens = [
            token for token in tokens
            if len(token) >= self.min_token_length and token.lower() not in self.java_stopwords
        ]

        return filtered_tokens

    def __call__(self, text: str) -> list[str]:
        """
        TfidfVectorizerのtokenizer引数で使用するためのメソッド
        LongNameとParentの両方に対応する汎用的なトークン化
        """
        if not text or pd.isna(text):
            return []

        text_str = str(text)

        # メソッドシグネチャかクラス名かを判定
        if '(' in text_str and ')' in text_str:
            # メソッドシグネチャとして処理
            return self.tokenize_longname(text_str)
        else:
            # クラス名として処理
            return self.tokenize_parent(text_str)


class SimplifiedBugHunter:
    """
    カスタムJavaトークナイザー統合版ダウンサンプリング版バグ予測クラス

    主な変更点:
    - JavaCodeTokenizerを使用したTF-IDF特徴量抽出
    - LongNameとParentカラムのJava固有の構造を考慮したトークン化
    - CamelCase、snake_case、パッケージ構造の適切な処理
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 1000,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False):
        """
        コンストラクタ

        Parameters:
            feature_selection_threshold (float): 特徴量重要度の閾値
            tfidf_max_features (int): TF-IDFで生成する特徴量の最大数
            java_tokenizer_min_length (int): Javaトークナイザーの最小トークン長
            include_package_tokens (bool): パッケージ名のトークンを含めるかどうか
        """
        self.best_model = None
        self.feature_importance = None
        self.all_feature_names = None
        self.selected_features = None
        self.optimization_history = []
        self.best_params: Dict[str, Any] = {}
        self.best_loss = float('inf')
        self.feature_selection_threshold = feature_selection_threshold
        self.initial_X = None
        self.initial_y = None
        self.tfidf_vectorizer_longname = None
        self.tfidf_vectorizer_parent = None
        self.tfidf_max_features = tfidf_max_features
        self.project_dummies_columns = None
        self.scaler = None

        # Javaトークナイザーの初期化
        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        # ダウンサンプリング関連の属性
        self.downsampled_X = None
        self.downsampled_y = None
        self.original_class_distribution = None
        self.downsampled_train_distribution = None

    def read_data(self, data_path: str) -> pd.DataFrame:
        """データ読み込み"""
        print("\n=== 1) データ読み込み ===")
        df = pd.read_csv(data_path)
        print(f"読み込み完了: {len(df)}行, {len(df.columns)}列")
        return df

    def prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> tuple:
        """データの前処理とラベル作成（カスタムJavaトークナイザー使用）"""
        print("\n=== データ前処理（カスタムJavaトークナイザー + TF-IDF + One-Hot Encoding + 正規化使用）===")

        # ラベル作成（Number of Bugs > 0.5で1、それ以外で0）
        y = (data["Number of Bugs"] > 0.5).astype(int) if "Number of Bugs" in data.columns and is_training else None

        # 元の数値特徴量を選択（Number of Bugsを除く）
        numerical_feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_feature_columns = [col for col in numerical_feature_columns if col != "Number of Bugs"]
        X_numerical = data[numerical_feature_columns].copy()

        # 欠損値・無限値の処理 (数値特徴量)
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
        X_numerical = X_numerical.fillna(0)

        # 正規化処理
        if is_training:
            self.scaler = StandardScaler()
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        else:
            if self.scaler is None:
                raise ValueError("Scalerが学習されていません。まず訓練データでprepare_dataを実行してください。")
            X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X_numerical.index)

        # LongNameカラムにカスタムJavaトークナイザーを使用したTF-IDFを適用
        longname_data = data['LongName'].fillna("").astype(str)
        if is_training:
            self.tfidf_vectorizer_longname = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                tokenizer=self.java_tokenizer,
                lowercase=False,  # トークナイザー内で小文字化を行うため
                token_pattern=None  # カスタムトークナイザーを使用するため
            )
            X_longname_tfidf = self.tfidf_vectorizer_longname.fit_transform(longname_data)
            print(f"LongName TF-IDF特徴量数: {X_longname_tfidf.shape[1]}")
        else:
            if self.tfidf_vectorizer_longname is None:
                raise ValueError("TF-IDF vectorizer (LongName)が学習されていません。まず訓練データでprepare_dataを実行してください。")
            X_longname_tfidf = self.tfidf_vectorizer_longname.transform(longname_data)
        X_longname_tfidf_df = pd.DataFrame(X_longname_tfidf.toarray(),
                                            columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf.shape[1])],
                                            index=longname_data.index)

        # ParentカラムにカスタムJavaトークナイザーを使用したTF-IDFを適用
        parent_data = data['Parent'].fillna("").astype(str)
        if is_training:
            self.tfidf_vectorizer_parent = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                tokenizer=self.java_tokenizer,
                lowercase=False,  # トークナイザー内で小文字化を行うため
                token_pattern=None  # カスタムトークナイザーを使用するため
            )
            X_parent_tfidf = self.tfidf_vectorizer_parent.fit_transform(parent_data)
            print(f"Parent TF-IDF特徴量数: {X_parent_tfidf.shape[1]}")
        else:
            if self.tfidf_vectorizer_parent is None:
                raise ValueError("TF-IDF vectorizer (Parent)が学習されていません。まず訓練データでprepare_dataを実行してください。")
            X_parent_tfidf = self.tfidf_vectorizer_parent.transform(parent_data)
        X_parent_tfidf_df = pd.DataFrame(X_parent_tfidf.toarray(),
                                          columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf.shape[1])],
                                          index=parent_data.index)

        # ProjectカラムにOne-Hot Encodingを適用
        project_data = data['Project'].fillna("Unknown").astype(str)
        X_project_onehot = pd.get_dummies(project_data, prefix='Project', dtype=int)
        if is_training:
            self.project_dummies_columns = X_project_onehot.columns.tolist()
        else:
            if self.project_dummies_columns is None:
                raise ValueError("One-Hot Encodingのカラム情報が学習されていません。まず訓練データでprepare_dataを実行してください。")
            X_project_onehot = X_project_onehot.reindex(columns=self.project_dummies_columns, fill_value=0)

        # 全てのXを結合
        X = pd.concat([X_numerical_scaled_df, X_longname_tfidf_df, X_parent_tfidf_df, X_project_onehot], axis=1)

        if is_training:
            self.all_feature_names = X.columns.tolist()
            self.initial_X = X
            self.initial_y = y

            print(f"初期使用特徴量数 (数値 + Java TF-IDF + One-Hot Encoding): {len(X.columns)}")
            print(f"  - 数値特徴量: {len(numerical_feature_columns)}")
            print(f"  - LongName TF-IDF: {X_longname_tfidf.shape[1]}")
            print(f"  - Parent TF-IDF: {X_parent_tfidf.shape[1]}")
            print(f"  - Project One-Hot: {len(X_project_onehot.columns)}")
            print(f"ラベル分布: 0={sum(y==0)}, 1={sum(y==1)}")

            # クラス分布の情報を保存
            self.original_class_distribution = {
                'class_0': sum(y==0),
                'class_1': sum(y==1),
                'total': len(y)
            }
            class_ratio = sum(y==1) / len(y)
            print(f"クラス分布: {class_ratio:.3f} (1の割合)")

            # サンプルトークン化結果の表示
            print("\n=== サンプルトークン化結果 ===")
            sample_longname = longname_data.iloc[0] if len(longname_data) > 0 else ""
            sample_parent = parent_data.iloc[0] if len(parent_data) > 0 else ""

            if sample_longname:
                sample_tokens_longname = self.java_tokenizer(sample_longname)
                print(f"LongName例: {sample_longname}")
                print(f"→ トークン: {sample_tokens_longname}")

            if sample_parent:
                sample_tokens_parent = self.java_tokenizer(sample_parent)
                print(f"Parent例: {sample_parent}")
                print(f"→ トークン: {sample_tokens_parent}")
        else:
            print(f"予測データの前処理完了: {len(X.columns)}列")

        return X, y

    def apply_downsampling(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        ダウンサンプリングを適用してクラス不均衡を解決

        Parameters:
            X (pd.DataFrame): 特徴量データ
            y (pd.Series): ラベルデータ

        Returns:
            tuple: ダウンサンプリング後の(X, y)
        """
        print("\n=== ダウンサンプリング適用 ===")

        # 各クラスのデータを分離
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]
        y_class_0 = y[y == 0]
        y_class_1 = y[y == 1]

        print(f"ダウンサンプリング前 - クラス0: {len(X_class_0)}件, クラス1: {len(X_class_1)}件")

        # 少数派クラス（クラス1）の数に合わせて多数派クラス（クラス0）をダウンサンプリング
        minority_class_size = len(X_class_1)

        if len(X_class_0) > minority_class_size:
            # ダウンサンプリングを実行
            X_class_0_downsampled, y_class_0_downsampled = resample(
                X_class_0, y_class_0,
                n_samples=minority_class_size,
                random_state=GLOBAL_SEED,
                replace=False
            )

            # ダウンサンプリング後のデータを結合
            X_balanced = pd.concat([X_class_0_downsampled, X_class_1], axis=0)
            y_balanced = pd.concat([y_class_0_downsampled, y_class_1], axis=0)

            # インデックスをリセット
            X_balanced = X_balanced.reset_index(drop=True)
            y_balanced = y_balanced.reset_index(drop=True)

            # ランダムにシャッフル
            shuffle_indices = np.random.RandomState(GLOBAL_SEED).permutation(len(X_balanced))
            X_balanced = X_balanced.iloc[shuffle_indices].reset_index(drop=True)
            y_balanced = y_balanced.iloc[shuffle_indices].reset_index(drop=True)

            print(f"ダウンサンプリング後 - クラス0: {len(X_class_0_downsampled)}件, クラス1: {len(X_class_1)}件")
            print(f"総データ数: {len(X_balanced)}件 (削減率: {(1 - len(X_balanced) / len(X)) * 100:.1f}%)")

        else:
            # ダウンサンプリングが不要な場合
            X_balanced = X.copy()
            y_balanced = y.copy()
            print("ダウンサンプリング不要: クラス1の方が多いか、同数です")

        # ダウンサンプリング後の訓練データのクラス分布を保存
        self.downsampled_train_distribution = {
            'class_0': sum(y_balanced == 0),
            'class_1': sum(y_balanced == 1),
            'total': len(y_balanced)
        }

        return X_balanced, y_balanced

    def log_loss_function(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Log Loss損失関数"""
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred_proba_clipped)

    def _evaluate_model_with_cv(self, model_instance: Any, X: pd.DataFrame, y: pd.Series, k_folds: int = 3) -> float:
        """任意のモデルインスタンスを交差検証で評価（Log Loss）"""
        try:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
            total_loss = 0.0

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model_instance.fit(X_train_fold, y_train_fold)
                y_pred_proba = model_instance.predict_proba(X_val_fold)[:, 1]
                fold_loss = self.log_loss_function(y_val_fold, y_pred_proba)
                total_loss += fold_loss

            return total_loss / k_folds

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')

    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 10) -> Dict[str, Any]:
        """Random ForestのLog Lossベース ベイジアン最適化"""
        print("--- Random Forestの最適化 ---")
        search_space = [
            Integer(100, 1000, name='n_estimators'),
            Integer(10, 20, name='max_depth'),
        ]

        def objective(params):
            n_estimators, max_depth = params
            rf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth),
                                        random_state=GLOBAL_SEED, n_jobs=-1)
            loss = self._evaluate_model_with_cv(rf, X, y)
            print(f"  RF Params: n_estimators={int(n_estimators)}, max_depth={int(max_depth)}, Loss: {loss:.4f}")
            return loss

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'n_estimators': int(result.x[0]),
            'max_depth': int(result.x[1])
        }
        print(f"Random Forest 最良損失: {result.fun:.4f}, 最良パラメータ: {optimal_params}")
        return optimal_params

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 10) -> Dict[str, Any]:
        """XGBoostのLog Lossベース ベイジアン最適化"""
        print("--- XGBoostの最適化 ---")
        search_space = [
            Integer(100, 1000, name='n_estimators'),
            Integer(10, 20, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree')
        ]

        def objective(params):
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree = params
            model = xgb.XGBClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=GLOBAL_SEED,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            loss = self._evaluate_model_with_cv(model, X, y)
            print(f"  XGB Params: n_estimators={int(n_estimators)}, max_depth={int(max_depth)}, "
                  f"learning_rate={learning_rate:.3f}, subsample={subsample:.3f}, "
                  f"colsample_bytree={colsample_bytree:.3f}, Loss: {loss:.4f}")
            return loss

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'n_estimators': int(result.x[0]),
            'max_depth': int(result.x[1]),
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4]
        }
        print(f"XGBoost 最良損失: {result.fun:.4f}, 最良パラメータ: {optimal_params}")
        return optimal_params

    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 15) -> Dict[str, Any]:
        """LightGBMのLog Lossベース ベイジアン最適化"""
        print("--- LightGBMの最適化 ---")
        search_space = [
            Integer(100, 1000, name='n_estimators'),
            Integer(5, 20, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Integer(10, 100, name='num_leaves')
        ]

        def objective(params):
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree, num_leaves = params
            model = lgb.LGBMClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                num_leaves=int(num_leaves),
                random_state=GLOBAL_SEED,
                n_jobs=-1,
                verbose=-1  # ログ出力を抑制
            )
            loss = self._evaluate_model_with_cv(model, X, y)
            print(f"  LGB Params: n_estimators={int(n_estimators)}, max_depth={int(max_depth)}, "
                  f"learning_rate={learning_rate:.3f}, subsample={subsample:.3f}, "
                  f"colsample_bytree={colsample_bytree:.3f}, num_leaves={int(num_leaves)}, Loss: {loss:.4f}")
            return loss

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'n_estimators': int(result.x[0]),
            'max_depth': int(result.x[1]),
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4],
            'num_leaves': int(result.x[5])
        }
        print(f"LightGBM 最良損失: {result.fun:.4f}, 最良パラメータ: {optimal_params}")
        return optimal_params

    def optimize_catboost(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 15) -> Dict[str, Any]:
        """CatBoostのLog Lossベース ベイジアン最適化"""
        print("--- CatBoostの最適化 ---")
        search_space = [
            Integer(100, 1000, name='iterations'),
            Integer(5, 12, name='depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 1.0, name='subsample')
        ]

        def objective(params):
            iterations, depth, learning_rate, subsample = params
            model = CatBoostClassifier(
                iterations=int(iterations),
                depth=int(depth),
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=GLOBAL_SEED,
                verbose=False,  # ログ出力を抑制
                thread_count=-1
            )
            loss = self._evaluate_model_with_cv(model, X, y)
            print(f"  CAT Params: iterations={int(iterations)}, depth={int(depth)}, "
                  f"learning_rate={learning_rate:.3f}, subsample={subsample:.3f}, Loss: {loss:.4f}")
            return loss

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'iterations': int(result.x[0]),
            'depth': int(result.x[1]),
            'learning_rate': result.x[2],
            'subsample': result.x[3]
        }
        print(f"CatBoost 最良損失: {result.fun:.4f}, 最良パラメータ: {optimal_params}")
        return optimal_params # This line was missing

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'iterations': int(result.x[0]),
            'depth': int(result.x[1]),
            'learning_rate': result.x[2],
            'subsample': result.x[3]
        }
    def optimize_knn(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 10) -> Dict[str, Any]:
        """KNNのLog Lossベース ベイジアン最適化"""
        print("--- KNNの最適化 ---")
        search_space = [
            Integer(10, 50, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Integer(1, 2, name='p') # p=1 for Manhattan, p=2 for Euclidean
        ]

        def objective(params):
            n_neighbors, weights, p = params
            model = KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights, p=int(p), n_jobs=-1)
            loss = self._evaluate_model_with_cv(model, X, y)
            print(f"  KNN Params: n_neighbors={int(n_neighbors)}, weights={weights}, p={int(p)}, Loss: {loss:.4f}")
            return loss

        result = gp_minimize(objective, search_space, n_calls=max_iterations, random_state=GLOBAL_SEED, acq_func='EI', n_initial_points=5)
        optimal_params = {
            'n_neighbors': int(result.x[0]),
            'weights': result.x[1],
            'p': int(result.x[2])
        }
        print(f"KNN 最良損失: {result.fun:.4f}, 最良パラメータ: {optimal_params}")
        return optimal_params

    def train_initial_model_for_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """特徴量重要度を取得するための初期モデルを学習（ダウンサンプリング済みデータで学習）"""
        print("\n=== 特徴量重要度取得のための初期モデル学習（カスタムJavaトークナイザー版）===")

        initial_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        initial_rf.fit(X, y)
        self.feature_importance = initial_rf.feature_importances_
        print("初期モデル学習と特徴量重要度の計算が完了しました。")
        return initial_rf

    def select_features_by_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """特徴量重要度に基づいて特徴量を選択"""
        if self.feature_importance is None:
            raise ValueError("特徴量重要度が計算されていません。まず初期モデルを学習してください。")

        feature_names = self.all_feature_names
        if len(feature_names) != len(self.feature_importance):
            print("警告: 特徴量名と重要度の数が一致しません。現在のXのカラムを再確認します。")
            feature_names = X.columns.tolist()
            if len(feature_names) != len(self.feature_importance):
                raise ValueError("特徴量名と重要度の数が一致しません。")

        # 重要度が閾値以上の特徴量を選択
        selected_feature_indices = np.where(self.feature_importance >= self.feature_selection_threshold)[0]
        self.selected_features = [feature_names[i] for i in selected_feature_indices]

        # 選択された特徴量が存在しない場合のエラーハンドリング
        if not self.selected_features:
            print(f"警告: 特徴量重要度の閾値 {self.feature_selection_threshold} では、どの特徴量も選択されませんでした。")
            print("すべての特徴量を再度含めて処理を続行します。")
            self.selected_features = feature_names
            X_selected = X
        else:
            X_selected = X[self.selected_features]

        print(f"\n=== 特徴量削減 ===")
        print(f"特徴量削減前の数: {len(feature_names)}")
        print(f"選択された特徴量数 (閾値 {self.feature_selection_threshold}): {len(self.selected_features)}")

        # 特徴量タイプ別の統計
        longname_count = len([f for f in self.selected_features if f.startswith('LongName_tfidf_')])
        parent_count = len([f for f in self.selected_features if f.startswith('Parent_tfidf_')])
        project_count = len([f for f in self.selected_features if f.startswith('Project_')])
        numerical_count = len(self.selected_features) - longname_count - parent_count - project_count

        print(f"  - 数値特徴量: {numerical_count}")
        print(f"  - LongName TF-IDF: {longname_count}")
        print(f"  - Parent TF-IDF: {parent_count}")
        print(f"  - Project One-Hot: {project_count}")

        return X_selected

    def comprehensive_evaluation(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """包括的評価（デフォルトしきい値0.5のみ使用）"""
        print("\n=== 包括的評価 ===")

        # 評価指標の計算
        results = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred),
            'LogLoss': self.log_loss_function(y_true, y_pred_proba),
            'Threshold': 0.5
        }

        print("\n--- 評価結果（しきい値=0.5）---")
        for metric, score in results.items():
            if metric != 'Threshold':
                print(f"{metric}: {score:.4f}")

        return results

    def get_feature_analysis(self) -> dict:
        """特徴量分析結果の取得"""
        params_to_return = self.best_params.copy() if self.best_params else {}

        # ダウンサンプリング情報を追加
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
            'feature_importance': self.feature_importance,
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
        """特徴量重要度テーブルの表示"""
        if self.feature_importance is None or self.selected_features is None:
            print("特徴量重要度は計算されていません。または利用可能なモデルから取得できませんでした。")
            return

        feature_names = self.selected_features
        if len(feature_names) != len(self.feature_importance):
            if self.all_feature_names is not None and len(self.feature_importance) == len(self.all_feature_names):
                feature_names = self.all_feature_names
            else:
                print("エラー: 選択された特徴量名と最終モデルの重要度の数が一致しません。")
                return

        importance_df = pd.DataFrame({
            '特徴量': feature_names,
            '重要度': self.feature_importance
        }).sort_values('重要度', ascending=False)

        # self.selected_features に含まれる特徴量のみをフィルタリング
        importance_df_filtered = importance_df[importance_df['特徴量'].isin(self.selected_features)]

        # 特徴量タイプの分類
        importance_df_filtered['タイプ'] = importance_df_filtered['特徴量'].apply(
            lambda x: 'LongName TF-IDF' if x.startswith('LongName_tfidf_')
                     else 'Parent TF-IDF' if x.startswith('Parent_tfidf_')
                     else 'Project' if x.startswith('Project_')
                     else '数値'
        )

        print(f"\n=== 上位{top_n}特徴量重要度 (カスタムJavaトークナイザー版) ===")
        display_df = importance_df_filtered.head(top_n)[['特徴量', 'タイプ', '重要度']].copy()
        print(display_df.to_string(index=False))

        # タイプ別の統計
        print(f"\n=== 特徴量タイプ別統計 ===")
        type_stats = importance_df_filtered['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_importance = importance_df_filtered[importance_df_filtered['タイプ'] == feature_type]['重要度'].mean()
            print(f"{feature_type}: {count}個 (平均重要度: {avg_importance:.4f})")

        return importance_df_filtered

    def display_downsampling_summary(self):
        """ダウンサンプリングのサマリー表示"""
        if not self.original_class_distribution or not self.downsampled_train_distribution:
            print("ダウンサンプリング情報がありません。")
            return

        print("\n=== ダウンサンプリング サマリー ===")
        print(f"元データ (全データ):")
        print(f"  クラス 0: {self.original_class_distribution['class_0']:,}件")
        print(f"  クラス 1: {self.original_class_distribution['class_1']:,}件")
        print(f"  合計: {self.original_class_distribution['total']:,}件")

        downsampled_class_0 = self.downsampled_train_distribution['class_0']
        downsampled_class_1 = self.downsampled_train_distribution['class_1']
        downsampled_total = self.downsampled_train_distribution['total']

        print(f"\nダウンサンプリング後 (訓練データ):")
        print(f"  クラス 0: {downsampled_class_0:,}件")
        print(f"  クラス 1: {downsampled_class_1:,}件")
        print(f"  合計: {downsampled_total:,}件")

        reduction_rate = (1 - downsampled_total / self.original_class_distribution['total']) * 100
        print(f"\n元データ全体からの削減率: {reduction_rate:.1f}%")

        original_imbalance = self.original_class_distribution['class_0'] / self.original_class_distribution['class_1'] if self.original_class_distribution['class_1'] > 0 else float('inf')
        new_imbalance = downsampled_class_0 / downsampled_class_1 if downsampled_class_1 > 0 else float('inf')
        print(f"クラス不均衡比 (クラス0/クラス1):")
        print(f"  元データ (全データ): {original_imbalance:.2f}:1")
        print(f"  ダウンサンプリング後 (訓練データ): {new_imbalance:.2f}:1")

    def display_tokenizer_analysis(self, sample_size: int = 5):
        """Javaトークナイザーの動作例を表示"""
        if self.initial_X is None:
            print("データが準備されていません。まずrun_pipeline()を実行してください。")
            return

        print("\n=== Javaトークナイザー動作例 ===")
        print(f"設定:")
        print(f"  最小トークン長: {self.java_tokenizer.min_token_length}")
        print(f"  パッケージトークン含む: {self.java_tokenizer.include_package_tokens}")
        print(f"  ストップワード数: {len(self.java_tokenizer.java_stopwords)}")


# ----------------------------------------------------------------------
# 改良版StackedBugHunter クラス
# ----------------------------------------------------------------------
class StackedBugHunter(SimplifiedBugHunter):
    """
    スタッキングアンサンブル版バグ予測クラス

    主要改良点:
    1. 予測時のベースモデル再学習問題を修正
    2. メタ特徴量生成の効率化
    3. 各ベースモデルの個別最適化とアンサンブル
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 1000,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 n_splits_stacking: int = 5):
        super().__init__(
            feature_selection_threshold=feature_selection_threshold,
            tfidf_max_features=tfidf_max_features,
            java_tokenizer_min_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )
        self.base_models: Dict[str, Any] = {}
        self.meta_model = None
        self.n_splits_stacking = n_splits_stacking
        self.base_model_optimal_params: Dict[str, Dict[str, Any]] = {}
        # 予測用の完全学習済みベースモデルを保存
        self.trained_base_models: Dict[str, Any] = {}

    def _train_base_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Any:
        """指定されたパラメータでベースモデルを学習して返す"""
        if model_name == 'rf':
            model = RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1, **params)
        elif model_name == 'xgb':
            model = xgb.XGBClassifier(
                random_state=GLOBAL_SEED,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                **params
            )
        elif model_name == 'knn':
            model = KNeighborsClassifier(n_jobs=-1, **params)
        elif model_name == 'lgb':
            model = lgb.LGBMClassifier(
                random_state=GLOBAL_SEED,
                n_jobs=-1,
                verbose=-1,
                **params
            )
        elif model_name == 'catboost':
            model = CatBoostClassifier(
                random_state=GLOBAL_SEED,
                verbose=False,
                thread_count=-1,
                **params
            )
        else:
            raise ValueError(f"不明なベースモデル名: {model_name}")

        model.fit(X, y)
        return model

    def _generate_meta_features(self, X_train_base: pd.DataFrame, y_train_base: pd.Series,
                              X_test_base: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        ベースモデルのOOF予測とテストセット予測を生成し、メタ特徴量とする

        Parameters:
            X_train_base (pd.DataFrame): ベースモデルの訓練データ（特徴量削減後）
            y_train_base (pd.Series): ベースモデルの訓練ラベル
            X_test_base (pd.DataFrame): ベースモデルのテストデータ（特徴量削減後）

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (X_meta_train, X_meta_test)
        """
        print(f"\n=== メタ特徴量生成 (K={self.n_splits_stacking} Fold Stacking) ===")
        kf = StratifiedKFold(n_splits=self.n_splits_stacking, shuffle=True, random_state=GLOBAL_SEED)

        # メタモデルの訓練データ (OOF予測)
        oof_preds = np.zeros((len(X_train_base), len(self.base_model_optimal_params)))
        # メタモデルのテストデータ (各ベースモデルの予測)
        test_preds = np.zeros((len(X_test_base), len(self.base_model_optimal_params)))

        # モデル名の順序を固定
        model_names = list(self.base_model_optimal_params.keys())

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_base, y_train_base)):
            print(f"--- Stacking Fold {fold + 1}/{self.n_splits_stacking} ---")
            X_fold_train, X_fold_val = X_train_base.iloc[train_idx], X_train_base.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_base.iloc[train_idx], y_train_base.iloc[val_idx]

            for i, model_name in enumerate(model_names):
                print(f"  学習中のベースモデル: {model_name}")

                # 各フォールドでベースモデルを学習
                trained_model = self._train_base_model(
                    model_name, X_fold_train, y_fold_train,
                    self.base_model_optimal_params[model_name]
                )

                # OOF予測
                oof_preds[val_idx, i] = trained_model.predict_proba(X_fold_val)[:, 1]

                # テストセット予測（各フォールドの予測を累積）
                test_preds[:, i] += trained_model.predict_proba(X_test_base)[:, 1] / self.n_splits_stacking

        # 予測用に全訓練データでベースモデルを再学習して保存
        print("\n=== 予測用ベースモデルの最終学習 ===")
        self.trained_base_models = {}
        for i, model_name in enumerate(model_names):
            print(f"  最終学習中: {model_name}")
            self.trained_base_models[model_name] = self._train_base_model(
                model_name, X_train_base, y_train_base,
                self.base_model_optimal_params[model_name]
            )

        X_meta_train = pd.DataFrame(
            oof_preds,
            columns=[f'{name}_pred' for name in model_names],
            index=X_train_base.index
        )
        X_meta_test = pd.DataFrame(
            test_preds,
            columns=[f'{name}_pred' for name in model_names],
            index=X_test_base.index
        )

        print(f"メタ訓練特徴量形状: {X_meta_train.shape}")
        print(f"メタテスト特徴量形状: {X_meta_test.shape}")
        return X_meta_train, X_meta_test

    def run_stacking_pipeline(self, data_path: str):
        """スタッキングアンサンブルパイプラインの実行"""
        print("\n" + "="*60)
        print("=== 改良版スタッキングアンサンブル バグ予測パイプライン ===")
        print("=== (RF + XGBoost + KNN + LightGBM + CatBoost + LogisticRegression) ===")
        print("="*60)

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備（カスタムJavaトークナイザー、TF-IDF、One-Hot Encoding、正規化）
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. ダウンサンプリングをここで一度だけ適用
        X_downsampled, y_downsampled = self.apply_downsampling(X_full, y_full)
        print(f"ダウンサンプリング後データセットサイズ: {len(X_downsampled)}行")

        # 4. データ分割（ダウンサンプリング後のデータに対して）
        X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(
            X_downsampled, y_downsampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_downsampled
        )
        print(f"訓練データ (ダウンサンプリング後): {len(X_train_ds)}行, テストデータ (ダウンサンプリング後): {len(X_test_ds)}行")

        # 5. 特徴量重要度取得のための初期モデル学習（ダウンサンプリング済み訓練データで学習）
        self.train_initial_model_for_feature_importance(X_train_ds, y_train_ds)

        # 6. 特徴量削減
        X_train_reduced = self.select_features_by_importance(X_train_ds)
        X_test_reduced = self.select_features_by_importance(X_test_ds)

        # 7. 各ベースモデルのハイパーパラメータ最適化（削減された訓練データで実施）
        print("\n=== 各ベースモデルのハイパーパラメータ最適化 ===")
        self.base_model_optimal_params['rf'] = self.optimize_random_forest(
            X_train_reduced, y_train_ds, max_iterations=15
        )
        self.base_model_optimal_params['xgb'] = self.optimize_xgboost(
            X_train_reduced, y_train_ds, max_iterations=15
        )
        self.base_model_optimal_params['knn'] = self.optimize_knn(
            X_train_reduced, y_train_ds, max_iterations=15
        )
        self.base_model_optimal_params['lgb'] = self.optimize_lightgbm(
            X_train_reduced, y_train_ds, max_iterations=15
        )
        self.base_model_optimal_params['catboost'] = self.optimize_catboost(
            X_train_reduced, y_train_ds, max_iterations=15
        )

        # 8. メタ特徴量の生成（ベースモデルのOOF予測）
        X_meta_train, X_meta_test = self._generate_meta_features(
            X_train_reduced, y_train_ds, X_test_reduced
        )

        # 9. メタモデル（ロジスティック回帰）の学習
        print("\n=== メタモデル（ロジスティック回帰）の学習 ===")
        self.meta_model = LogisticRegression(
            solver='liblinear',
            random_state=GLOBAL_SEED,
            max_iter=1000
        )
        self.meta_model.fit(X_meta_train, y_train_ds)
        print("メタモデル学習完了")

        # 10. 個別ベースモデルの評価
        print("\n=== 個別ベースモデルの評価 ===")
        base_model_results = {}

        for model_name in self.base_model_optimal_params.keys():
            print(f"\n--- {model_name.upper()}モデル単体評価 ---")
            # 最適化されたパラメータで単体モデルを学習
            single_model = self._train_base_model(
                model_name, X_train_reduced, y_train_ds,
                self.base_model_optimal_params[model_name]
            )

            y_pred_proba_single = single_model.predict_proba(X_test_reduced)[:, 1]
            y_pred_single = (y_pred_proba_single >= 0.5).astype(int)

            single_results = self.comprehensive_evaluation(
                y_test_ds, y_pred_single, y_pred_proba_single
            )
            base_model_results[model_name] = single_results

        # 11. スタッキングモデルの最終評価
        print("\n=== スタッキングモデルの最終評価 ===")
        y_pred_proba_stacked = self.meta_model.predict_proba(X_meta_test)[:, 1]
        y_pred_stacked = (y_pred_proba_stacked >= 0.5).astype(int)

        stacked_results = self.comprehensive_evaluation(
            y_test_ds, y_pred_stacked, y_pred_proba_stacked
        )

        # 12. 結果比較の表示
        print("\n" + "="*60)
        print("=== モデル性能比較サマリー ===")
        print("="*60)

        all_results = {**base_model_results, 'Stacking': stacked_results}
        comparison_df = pd.DataFrame({
            model_name: {
                'F1': results['F1'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'Accuracy': results['Accuracy'],
                'LogLoss': results['LogLoss']
            }
            for model_name, results in all_results.items()
        }).round(4)

        print(comparison_df.T.to_string())

        # 最高性能の特定
        best_f1_model = comparison_df.loc['F1'].idxmax()
        best_logloss_model = comparison_df.loc['LogLoss'].idxmin()

        print(f"\n最高F1スコア: {best_f1_model} ({comparison_df.loc['F1', best_f1_model]:.4f})")
        print(f"最低LogLoss: {best_logloss_model} ({comparison_df.loc['LogLoss', best_logloss_model]:.4f})")

        return stacked_results, self.base_model_optimal_params, base_model_results

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """学習済みスタッキングモデルで予測を行う（改良版・列名チェック付き）"""
        if self.meta_model is None or not self.trained_base_models:
            raise ValueError("モデルが学習されていません。まずrun_stacking_pipeline()を実行してください。")
        if self.selected_features is None:
            raise ValueError("特徴量削減が実行されていません。")
        if any(v is None for v in [self.tfidf_vectorizer_longname, self.tfidf_vectorizer_parent,
                                  self.project_dummies_columns, self.scaler]):
            raise ValueError("前処理コンポーネントが学習されていません。")

        print(f"予測データの列名: {list(X.columns)}")

        # 必要な列が存在するかチェック
        required_text_columns = ['LongName', 'Parent', 'Project']
        missing_text_columns = [col for col in required_text_columns if col not in X.columns]
        if missing_text_columns:
            raise ValueError(f"必要な列が不足しています: {missing_text_columns}")

        # 予測データの前処理
        numerical_feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if "Number of Bugs" in numerical_feature_columns:
            numerical_feature_columns.remove("Number of Bugs")

        # 学習時に使用した数値列名を取得（scalerの feature_names_in_ を使用）
        if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
            trained_numerical_columns = list(self.scaler.feature_names_in_)
            print(f"学習時の数値列: {trained_numerical_columns}")

            # 予測データから学習時の数値列のみを選択
            available_numerical_columns = [col for col in trained_numerical_columns if col in X.columns]
            missing_numerical_columns = [col for col in trained_numerical_columns if col not in X.columns]

            if missing_numerical_columns:
                print(f"警告: 予測データに不足している数値列: {missing_numerical_columns}")
                print("不足している列は0で補完されます。")

            # 学習時の列順序に合わせてデータフレームを作成
            X_numerical = pd.DataFrame(0, index=X.index, columns=trained_numerical_columns)
            for col in available_numerical_columns:
                X_numerical[col] = X[col]
        else:
            # フォールバック: 現在の数値列をそのまま使用
            X_numerical = X[numerical_feature_columns].copy()
            print(f"予測データの数値列: {numerical_feature_columns}")

        # 欠損値・無限値の処理
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
        X_numerical = X_numerical.fillna(0)

        # 正規化
        try:
            X_numerical_scaled = self.scaler.transform(X_numerical)
            X_numerical_scaled_df = pd.DataFrame(
                X_numerical_scaled, columns=X_numerical.columns, index=X_numerical.index
            )
        except Exception as e:
            print(f"正規化エラー: {e}")
            print(f"学習時の特徴量数: {len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else '不明'}")
            print(f"予測時の特徴量数: {len(X_numerical.columns)}")
            raise

        # TF-IDF特徴量
        longname_data_pred = X['LongName'].fillna("").astype(str)
        X_longname_tfidf_pred = self.tfidf_vectorizer_longname.transform(longname_data_pred)
        X_longname_tfidf_df_pred = pd.DataFrame(
            X_longname_tfidf_pred.toarray(),
            columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf_pred.shape[1])],
            index=X.index
        )

        parent_data_pred = X['Parent'].fillna("").astype(str)
        X_parent_tfidf_pred = self.tfidf_vectorizer_parent.transform(parent_data_pred)
        X_parent_tfidf_df_pred = pd.DataFrame(
            X_parent_tfidf_pred.toarray(),
            columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf_pred.shape[1])],
            index=X.index
        )

        # One-Hot Encoding
        project_data_pred = X['Project'].fillna("Unknown").astype(str)
        X_project_onehot_pred = pd.get_dummies(project_data_pred, prefix='Project', dtype=int)
        X_project_onehot_pred = X_project_onehot_pred.reindex(columns=self.project_dummies_columns, fill_value=0)

        # 全特徴量の結合
        X_processed_full = pd.concat([
            X_numerical_scaled_df, X_longname_tfidf_df_pred,
            X_parent_tfidf_df_pred, X_project_onehot_pred
        ], axis=1)

        # 学習時に選択された特徴量のみを選択
        X_processed = pd.DataFrame(0, index=X_processed_full.index, columns=self.selected_features)
        for col in self.selected_features:
            if col in X_processed_full.columns:
                X_processed[col] = X_processed_full[col]

        print(f"最終予測用特徴量数: {len(X_processed.columns)}")

        # 学習済みベースモデルによるメタ特徴量の生成
        meta_features_predict = np.zeros((len(X_processed), len(self.trained_base_models)))

        for i, (model_name, trained_model) in enumerate(self.trained_base_models.items()):
            meta_features_predict[:, i] = trained_model.predict_proba(X_processed)[:, 1]

        # メタモデルによる最終予測
        meta_features_df = pd.DataFrame(
            meta_features_predict,
            columns=[f'{name}_pred' for name in self.trained_base_models.keys()],
            index=X_processed.index
        )

        y_pred_proba = self.meta_model.predict_proba(meta_features_df)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred, y_pred_proba

    def predict_with_base_models(self, X: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """各ベースモデル + スタッキングモデルの予測を個別に取得（改良版）"""
        if not self.trained_base_models or self.meta_model is None:
            raise ValueError("モデルが学習されていません。")

        # 必要な列が存在するかチェック
        required_text_columns = ['LongName', 'Parent', 'Project']
        missing_text_columns = [col for col in required_text_columns if col not in X.columns]
        if missing_text_columns:
            raise ValueError(f"必要な列が不足しています: {missing_text_columns}")

        # データ前処理（predict()と同じロジック）
        numerical_feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if "Number of Bugs" in numerical_feature_columns:
            numerical_feature_columns.remove("Number of Bugs")

        # 学習時に使用した数値列名を取得
        if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
            trained_numerical_columns = list(self.scaler.feature_names_in_)

            # 予測データから学習時の数値列のみを選択
            available_numerical_columns = [col for col in trained_numerical_columns if col in X.columns]
            missing_numerical_columns = [col for col in trained_numerical_columns if col not in X.columns]

            if missing_numerical_columns:
                print(f"警告: 予測データに不足している数値列: {missing_numerical_columns}")

            # 学習時の列順序に合わせてデータフレームを作成
            X_numerical = pd.DataFrame(0, index=X.index, columns=trained_numerical_columns)
            for col in available_numerical_columns:
                X_numerical[col] = X[col]
        else:
            X_numerical = X[numerical_feature_columns].copy()

        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
        X_numerical = X_numerical.fillna(0)

        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(
            X_numerical_scaled, columns=X_numerical.columns, index=X_numerical.index
        )

        longname_data_pred = X['LongName'].fillna("").astype(str)
        X_longname_tfidf_pred = self.tfidf_vectorizer_longname.transform(longname_data_pred)
        X_longname_tfidf_df_pred = pd.DataFrame(
            X_longname_tfidf_pred.toarray(),
            columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf_pred.shape[1])],
            index=X.index
        )

        parent_data_pred = X['Parent'].fillna("").astype(str)
        X_parent_tfidf_pred = self.tfidf_vectorizer_parent.transform(parent_data_pred)
        X_parent_tfidf_df_pred = pd.DataFrame(
            X_parent_tfidf_pred.toarray(),
            columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf_pred.shape[1])],
            index=X.index
        )

        project_data_pred = X['Project'].fillna("Unknown").astype(str)
        X_project_onehot_pred = pd.get_dummies(project_data_pred, prefix='Project', dtype=int)
        X_project_onehot_pred = X_project_onehot_pred.reindex(columns=self.project_dummies_columns, fill_value=0)

        X_processed_full = pd.concat([
            X_numerical_scaled_df, X_longname_tfidf_df_pred,
            X_parent_tfidf_df_pred, X_project_onehot_pred
        ], axis=1)

        X_processed = pd.DataFrame(0, index=X_processed_full.index, columns=self.selected_features)
        for col in self.selected_features:
            if col in X_processed_full.columns:
                X_processed[col] = X_processed_full[col]

        # 各モデルの予測結果を格納
        predictions = {}

        # ベースモデル個別の予測
        for model_name, trained_model in self.trained_base_models.items():
            y_pred_proba_base = trained_model.predict_proba(X_processed)[:, 1]
            y_pred_base = (y_pred_proba_base >= 0.5).astype(int)
            predictions[model_name] = (y_pred_base, y_pred_proba_base)

        # スタッキングモデルの予測
        meta_features_predict = np.zeros((len(X_processed), len(self.trained_base_models)))
        for i, (model_name, trained_model) in enumerate(self.trained_base_models.items()):
            meta_features_predict[:, i] = trained_model.predict_proba(X_processed)[:, 1]

        meta_features_df = pd.DataFrame(
            meta_features_predict,
            columns=[f'{name}_pred' for name in self.trained_base_models.keys()],
            index=X_processed.index
        )

        y_pred_proba_stacked = self.meta_model.predict_proba(meta_features_df)[:, 1]
        y_pred_stacked = (y_pred_proba_stacked >= 0.5).astype(int)
        predictions['stacking'] = (y_pred_stacked, y_pred_proba_stacked)

        return predictions

    def get_meta_model_coefficients(self) -> pd.DataFrame:
        """メタモデル（ロジスティック回帰）の係数を取得"""
        if self.meta_model is None:
            raise ValueError("メタモデルが学習されていません。")

        feature_names = [f'{name}_pred' for name in self.trained_base_models.keys()]
        coefficients_df = pd.DataFrame({
            'ベースモデル': list(self.trained_base_models.keys()),
            '係数': self.meta_model.coef_[0],
            '重要度': np.abs(self.meta_model.coef_[0])
        }).sort_values('重要度', ascending=False)

        return coefficients_df

    def display_stacking_analysis(self):
        """スタッキングアンサンブルの分析結果を表示"""
        print("\n=== スタッキングアンサンブル分析 ===")

        # メタモデルの係数表示
        try:
            coef_df = self.get_meta_model_coefficients()
            print("\n--- メタモデル（ロジスティック回帰）の係数 ---")
            print(coef_df.to_string(index=False))

            print(f"\n切片: {self.meta_model.intercept_[0]:.4f}")

            # 各ベースモデルの貢献度
            total_abs_coef = coef_df['重要度'].sum()
            coef_df['貢献度(%)'] = (coef_df['重要度'] / total_abs_coef * 100).round(2)
            print("\n--- ベースモデル貢献度 ---")
            for _, row in coef_df.iterrows():
                print(f"{row['ベースモデル']}: {row['貢献度(%)']}%")

        except Exception as e:
            print(f"メタモデル分析でエラーが発生しました: {e}")

        # ベースモデルのパラメータ表示
        print("\n--- 最適化されたベースモデルパラメータ ---")
        for model_name, params in self.base_model_optimal_params.items():
            print(f"\n{model_name.upper()}:")
            for param_name, param_value in params.items():
                if isinstance(param_value, float):
                    print(f"  {param_name}: {param_value:.4f}")
                else:
                    print(f"  {param_name}: {param_value}")


# 使用例とメイン実行部分
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # 改良版スタッキングアンサンブル版バグハンターのインスタンス作成
    bug_hunter = StackedBugHunter(
        feature_selection_threshold=0.001,
        tfidf_max_features=1000,
        java_tokenizer_min_length=3,
        include_package_tokens=False,  # パッケージ名を除外して、より重要な部分に集中
        n_splits_stacking=5  # スタッキングのK-Fold数
    )

    # スタッキングパイプライン実行
    stacked_results, optimal_base_params, base_model_results = bug_hunter.run_stacking_pipeline(data_path)

    print("\n" + "="*60)
    print("=== 拡張スタッキングアンサンブル バグ予測完了! ===")
    print("=== (RF + XGBoost + KNN + LightGBM + CatBoost) ===")
    print("="*60)
    print(f"スタッキング F1スコア: {stacked_results['F1']:.4f}")
    print(f"スタッキング Precision: {stacked_results['Precision']:.4f}")
    print(f"スタッキング Recall: {stacked_results['Recall']:.4f}")
    print(f"スタッキング Accuracy: {stacked_results['Accuracy']:.4f}")
    print(f"スタッキング LogLoss: {stacked_results['LogLoss']:.4f}")

    # 各種分析の表示
    bug_hunter.display_downsampling_summary()
    bug_hunter.display_feature_importance_table(top_n=15)
    bug_hunter.display_stacking_analysis()

    # 予測の例 (実際のデータから数行取得してテスト)
    print("\n=== 予測の実行例 ===")

    # 実際のデータから数行を読み込んでテスト用に使用
    test_data_full = pd.read_csv(data_path)
    print(f"実際のデータの列名: {list(test_data_full.columns)}")

    # 最初の3行をテストデータとして使用（Number of Bugsを除去）
    dummy_data = test_data_full.head(3).copy()
    if 'Number of Bugs' in dummy_data.columns:
        dummy_data = dummy_data.drop('Number of Bugs', axis=1)

    print(f"テストデータ形状: {dummy_data.shape}")
    print(f"テストデータの列: {list(dummy_data.columns)}")

    # スタッキングモデルの予測
    dummy_pred, dummy_proba = bug_hunter.predict(dummy_data)
    print(f"\nスタッキングモデル予測結果:")
    print(f"  クラス: {dummy_pred}")
    print(f"  確率: {dummy_proba.round(4)}")

    # 各ベースモデル + スタッキングの予測比較
    all_predictions = bug_hunter.predict_with_base_models(dummy_data)
    print(f"\n各モデルの予測比較:")
    for model_name, (pred_class, pred_proba) in all_predictions.items():
        print(f"  {model_name.upper()}: クラス={pred_class}, 確率={pred_proba.round(4)}")

    print("\n" + "="*60)
    print("実行完了!")
    print("="*60)
