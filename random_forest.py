"""
BugHunter: 論文に基づくバグ予測システム（RandomForest Feature Importance版）

論文: "An automatically created novel bug dataset and its validation in bug prediction"
著者: Rudolf Ferenc, Péter Gyimesi, Gábor Gyimesi, Zoltán Tóth, Tibor Gyimóthy
雑誌: Journal of Systems and Software, vol. 169, p. 110691, November 2020

特徴:
- 10分割交差検証による論文準拠の評価
- RandomUnderSamplerによるクラス不均衡対応
- RandomForest Feature Importanceによる特徴量選択
- カスタムJavaトークナイザーによるTF-IDF
- RandomForestによるバグ予測
- 最終モデルでのテストデータ評価
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import re
from typing import Set
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

    def __init__(self, min_token_length: int = 3, include_package_tokens: bool = True):
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


class BugHunter:
    """
    BugHunterデータセットに基づくバグ予測システム（RandomForest Feature Importance版）

    論文の手法を忠実に再現:
    - 10分割交差検証
    - RandomForestによる予測
    - RandomUnderSamplerによるクラス不均衡対応
    - RandomForest Feature Importanceによる特徴量選択
    - カスタムJavaトークナイザーによるTF-IDF
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 test_size: float = 0.2):
        """
        コンストラクタ

        Parameters:
            feature_selection_threshold (float): Feature Importanceによる特徴量選択の閾値
            tfidf_max_features (int): TF-IDFで生成する特徴量の最大数
            java_tokenizer_min_length (int): Javaトークナイザーの最小トークン長
            include_package_tokens (bool): パッケージ名のトークンを含めるかどうか
            test_size (float): テストデータの比率
        """
        # モデル関連
        self.best_model = None
        self.feature_importance = None

        # 特徴量関連
        self.all_feature_names = None
        self.selected_features = None
        self.feature_selection_threshold = feature_selection_threshold
        self.feature_importance_scores = None

        # 前処理関連
        self.initial_X = None
        self.initial_y = None
        self.tfidf_vectorizer_longname = None
        self.tfidf_vectorizer_parent = None
        self.tfidf_max_features = tfidf_max_features
        self.project_dummies_columns = None
        self.scaler = None

        # データ分割関連
        self.test_size = test_size
        self.test_results = None

        # Javaトークナイザーの初期化
        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        # サンプリング関連
        self.original_class_distribution = None
        self.resampled_train_distribution = None

        # 論文に基づくデフォルトパラメータ
        self.default_rf_params = {
            'n_estimators': 500,
            'max_depth': 30,
            'random_state': GLOBAL_SEED,
            'n_jobs': -1
        }

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

    def apply_undersampling(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        RandomUnderSamplerを適用してクラス不均衡を解決

        Parameters:
            X (pd.DataFrame): 特徴量データ
            y (pd.Series): ラベルデータ

        Returns:
            tuple: アンダーサンプリング後の(X, y)
        """
        print(f"サンプリング前 - クラス0: {sum(y==0)}件, クラス1: {sum(y==1)}件", end=" → ")

        try:
            sampler = RandomUnderSampler(random_state=GLOBAL_SEED)
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            y_balanced = pd.Series(y_resampled)

            print(f"クラス0: {sum(y_balanced==0)}件, クラス1: {sum(y_balanced==1)}件")

            # アンダーサンプリング後の訓練データのクラス分布を保存
            self.resampled_train_distribution = {
                'class_0': sum(y_balanced == 0),
                'class_1': sum(y_balanced == 1),
                'total': len(y_balanced)
            }
        except Exception as e:
            print(f"RandomUnderSamplerでエラーが発生しました: {e}")
            print("元のデータを使用して続行します。")
            X_balanced = X.copy()
            y_balanced = y.copy()

            self.resampled_train_distribution = {
                'class_0': sum(y_balanced == 0),
                'class_1': sum(y_balanced == 1),
                'total': len(y_balanced)
            }

        return X_balanced, y_balanced

    def select_features_by_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """RandomForest Feature Importanceに基づいて特徴量を選択"""
        print(f"\n=== RandomForest Feature Importanceによる特徴量選択 ===")
        print(f"特徴量選択前の数: {len(X.columns)}")

        # 特徴量選択用のRandomForestモデルを作成
        rf_selector = RandomForestClassifier(**self.default_rf_params)

        try:
            # モデルを学習してfeature importanceを取得
            rf_selector.fit(X, y)
            self.feature_importance_scores = rf_selector.feature_importances_

            # 閾値以上の重要度を持つ特徴量を選択
            selected_indices = np.where(self.feature_importance_scores >= self.feature_selection_threshold)[0]

            # 特徴量が1つも選択されない場合は上位50%を選択
            if len(selected_indices) == 0:
                print(f"閾値 {self.feature_selection_threshold} で特徴量が選択されませんでした。")
                print("上位50%の特徴量を選択します。")
                n_features_to_select = max(1, len(X.columns) // 2)
                selected_indices = np.argsort(self.feature_importance_scores)[-n_features_to_select:]

            self.selected_features = [X.columns[i] for i in selected_indices]

            # 選択された特徴量でDataFrameを作成
            X_selected_df = X[self.selected_features].copy()

            print(f"選択された特徴量数 (重要度≥{self.feature_selection_threshold}): {len(self.selected_features)}")

            # 特徴量タイプ別の統計
            longname_count = len([f for f in self.selected_features if f.startswith('LongName_tfidf_')])
            parent_count = len([f for f in self.selected_features if f.startswith('Parent_tfidf_')])
            project_count = len([f for f in self.selected_features if f.startswith('Project_')])
            numerical_count = len(self.selected_features) - longname_count - parent_count - project_count

            print(f"  - 数値特徴量: {numerical_count}")
            print(f"  - LongName TF-IDF: {longname_count}")
            print(f"  - Parent TF-IDF: {parent_count}")
            print(f"  - Project One-Hot: {project_count}")

            # 上位特徴量の表示
            print("\nRandomForest Feature Importance 上位5:")
            top_indices = np.argsort(self.feature_importance_scores)[-5:][::-1]
            for i, idx in enumerate(top_indices):
                feature_name = X.columns[idx]
                importance = self.feature_importance_scores[idx]
                selected = "✓" if feature_name in self.selected_features else "✗"
                print(f"  {i+1}. {feature_name}: {importance:.4f} {selected}")

        except Exception as e:
            print(f"RandomForest Feature Importanceによる特徴量選択でエラーが発生しました: {e}")
            print("全ての特徴量を使用して続行します。")
            self.selected_features = X.columns.tolist()
            self.feature_importance_scores = np.ones(len(X.columns)) / len(X.columns)
            X_selected_df = X.copy()

        return X_selected_df

    def evaluate_model_with_cv(self, X: pd.DataFrame, y: pd.Series, k_folds: int = 10) -> dict:
        """論文と同じ10分割交差検証を用いたRandomForestモデル評価"""
        print(f"\n=== {k_folds}分割交差検証による評価 ===")

        # Random Forestモデルの作成（デフォルトパラメータ使用）
        rf = RandomForestClassifier(**self.default_rf_params)

        # 交差検証の設定
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)

        # 評価指標の設定
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }

        print(f"交差検証設定:")
        print(f"  分割数: {k_folds}")
        print(f"  モデル: RandomForest")
        print(f"  パラメータ: {self.default_rf_params}")
        print(f"  データサイズ: {len(X)}行 × {len(X.columns)}列")

        # 各フォールドで独立にアンダーサンプリングを適用する交差検証
        fold_results = {metric: [] for metric in scoring.keys()}

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"フォールド {fold_idx + 1}/{k_folds} 処理中...", end=" ")

            # 元データから訓練・検証フォールドを作成
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # 訓練フォールドのみにアンダーサンプリングを適用
            X_train_fold_resampled, y_train_fold_resampled = self.apply_undersampling(
                X_train_fold, y_train_fold
            )

            # モデル学習（アンダーサンプリング適用済み訓練データ）
            rf_fold = RandomForestClassifier(**self.default_rf_params)
            rf_fold.fit(X_train_fold_resampled, y_train_fold_resampled)

            # 検証（元の検証データ）
            y_pred = rf_fold.predict(X_val_fold)
            y_pred_proba = rf_fold.predict_proba(X_val_fold)[:, 1]

            # 各指標を計算
            fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_results['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            fold_results['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_results['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))

            print(f"F1: {fold_results['f1'][-1]:.3f}")

        # 結果の統計計算
        cv_results = {}
        for metric in scoring.keys():
            scores = fold_results[metric]
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores

        return cv_results

    def evaluate_on_test_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """テストデータでの最終モデル評価"""
        print("\n=== テストデータでの最終モデル評価 ===")

        if self.best_model is None:
            raise ValueError("最終モデルが学習されていません。まずtrain_final_model()を実行してください。")

        # テストデータで予測
        y_test_pred = self.best_model.predict(X_test)
        y_test_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # 評価指標を計算
        test_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }

        # クラス分布の確認
        test_class_distribution = {
            'class_0': sum(y_test == 0),
            'class_1': sum(y_test == 1),
            'total': len(y_test)
        }

        # 予測結果の分布
        pred_class_distribution = {
            'predicted_class_0': sum(y_test_pred == 0),
            'predicted_class_1': sum(y_test_pred == 1)
        }

        print(f"テストデータサイズ: {len(X_test)}件")
        print(f"テストデータのクラス分布:")
        print(f"  クラス 0: {test_class_distribution['class_0']}件")
        print(f"  クラス 1: {test_class_distribution['class_1']}件")
        print(f"予測結果の分布:")
        print(f"  予測クラス 0: {pred_class_distribution['predicted_class_0']}件")
        print(f"  予測クラス 1: {pred_class_distribution['predicted_class_1']}件")

        print(f"\n=== テストデータ評価結果 ===")
        print(f"F1スコア: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"ROC-AUC: {test_results['roc_auc']:.4f}")

        # 結果を保存
        self.test_results = {
            **test_results,
            'test_class_distribution': test_class_distribution,
            'pred_class_distribution': pred_class_distribution
        }

        return self.test_results

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        """全データでRandomForestモデルを学習"""
        print("\n=== 最終RandomForestモデル学習 ===")

        # 全データにアンダーサンプリングを適用
        X_resampled, y_resampled = self.apply_undersampling(X, y)

        # RandomForestモデルの作成と学習
        self.best_model = RandomForestClassifier(**self.default_rf_params)
        self.best_model.fit(X_resampled, y_resampled)

        print("最終RandomForestモデル学習完了")
        print(f"パラメータ: {self.default_rf_params}")
        print(f"学習データ: {len(X_resampled)}件")

        # 特徴量重要度の取得
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
            print("特徴量重要度を取得しました。")

            # 上位5特徴量の表示
            if len(self.feature_importance) > 0 and len(self.selected_features) == len(self.feature_importance):
                top_indices = self.feature_importance.argsort()[-5:][::-1]
                print("最終モデル特徴量重要度 上位5:")
                for i, idx in enumerate(top_indices):
                    feature_name = self.selected_features[idx]
                    importance = self.feature_importance[idx]
                    print(f"  {i+1}. {feature_name}: {importance:.4f}")

        return self.best_model

    def predict(self, X: pd.DataFrame) -> tuple:
        """予測（学習済みモデルを使用）"""
        if self.best_model is None:
            raise ValueError("モデルが学習されていません。まずrun_pipeline()を実行してください。")
        if self.selected_features is None:
            raise ValueError("特徴量選択が実行されていません。まずrun_pipeline()を実行してください。")
        if self.tfidf_vectorizer_longname is None or self.tfidf_vectorizer_parent is None:
            raise ValueError("TF-IDF Vectorizerが学習されていません。まずrun_pipeline()を実行してください。")
        if self.project_dummies_columns is None:
            raise ValueError("One-Hot Encoderが学習されていません。まずrun_pipeline()を実行してください。")
        if self.scaler is None:
            raise ValueError("Scalerが学習されていません。まずrun_pipeline()を実行してください。")

        # 数値特徴量の準備（学習済みスケーラーを適用）
        numerical_feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if "Number of Bugs" in numerical_feature_columns:
            numerical_feature_columns.remove("Number of Bugs")

        X_numerical = X[numerical_feature_columns].copy()
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
        X_numerical = X_numerical.fillna(0)

        # 学習済みスケーラーを使用して数値特徴量を変換
        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X_numerical.index)

        # 学習済みTF-IDFベクトライザーを使用してTF-IDF変換を適用
        longname_data_pred = X['LongName'].fillna("").astype(str)
        X_longname_tfidf_pred = self.tfidf_vectorizer_longname.transform(longname_data_pred)
        X_longname_tfidf_df_pred = pd.DataFrame(X_longname_tfidf_pred.toarray(),
                                                  columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf_pred.shape[1])],
                                                  index=X.index)

        parent_data_pred = X['Parent'].fillna("").astype(str)
        X_parent_tfidf_pred = self.tfidf_vectorizer_parent.transform(parent_data_pred)
        X_parent_tfidf_df_pred = pd.DataFrame(X_parent_tfidf_pred.toarray(),
                                                columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf_pred.shape[1])],
                                                index=X.index)

        # 学習済みOne-Hot Encodingカラムを使用してOne-Hot Encoding変換を適用
        project_data_pred = X['Project'].fillna("Unknown").astype(str)
        X_project_onehot_pred = pd.get_dummies(project_data_pred, prefix='Project', dtype=int)
        X_project_onehot_pred = X_project_onehot_pred.reindex(columns=self.project_dummies_columns, fill_value=0)

        # 予測用の全特徴量を結合
        X_processed_full = pd.concat([X_numerical_scaled_df, X_longname_tfidf_df_pred, X_parent_tfidf_df_pred, X_project_onehot_pred], axis=1)

        # 学習時に選択された特徴量のみを選択
        X_processed_df = X_processed_full[self.selected_features]

        # 予測確率取得
        y_pred_proba = self.best_model.predict_proba(X_processed_df)[:, 1]

        # デフォルトしきい値での予測
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred, y_pred_proba

    def run_pipeline(self, data_path: str):
        """BugHunter用パイプライン（10分割交差検証 + テストデータ評価 + RandomForest Feature Importance版）"""
        print("=== BugHunter 10分割交差検証バグ予測パイプライン (RandomForest Feature Importance版) ===")
        print("特徴:")
        print("- 論文と同じ10分割交差検証を使用")
        print("- 各フォールドで独立にRandomUnderSamplerを適用")
        print("- RandomForest Feature Importanceによる特徴量選択を使用")
        print("- 標準的なRandomForestパラメータを使用")
        print("- 最終モデルのテストデータでの評価を実施")

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. 訓練・テストデータの分割
        print(f"\n=== 訓練・テストデータ分割 (テスト比率: {self.test_size}) ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=self.test_size,
            random_state=GLOBAL_SEED,
            stratify=y_full
        )

        print(f"訓練データ: {len(X_train)}件 (クラス0: {sum(y_train==0)}, クラス1: {sum(y_train==1)})")
        print(f"テストデータ: {len(X_test)}件 (クラス0: {sum(y_test==0)}, クラス1: {sum(y_test==1)})")

        # 4. RandomForest Feature Importanceによる特徴量選択（訓練データで特徴量選択器を学習）
        print("\n=== 特徴量選択器の学習（訓練データのみ使用） ===")
        # 一時的にアンダーサンプリングを適用して特徴量選択器を学習
        X_train_temp_resampled, y_train_temp_resampled = self.apply_undersampling(X_train, y_train)
        X_reduced = self.select_features_by_rf_importance(X_train_temp_resampled, y_train_temp_resampled)

        # 訓練・テストデータに特徴量選択を適用
        X_train_reduced_df = X_train[self.selected_features]
        X_test_reduced_df = X_test[self.selected_features]

        # 5. 10分割交差検証による評価（訓練データのみ使用）
        print("\n=== 交差検証評価（訓練データのみ使用） ===")
        cv_results = self.evaluate_model_with_cv(X_train_reduced_df, y_train, k_folds=10)

        # 6. 最終モデル学習（訓練データで）
        final_model = self.train_final_model(X_train_reduced_df, y_train)

        # 7. テストデータでの評価
        test_results = self.evaluate_on_test_data(X_test_reduced_df, y_test)

        # 8. 結果のまとめ
        print("\n=== 10分割交差検証結果（訓練データ） ===")
        print(f"F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        print(f"Recall: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        print(f"Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")

        print(f"\n=== 最終テストデータ評価結果 ===")
        print(f"F1スコア: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"ROC-AUC: {test_results['roc_auc']:.4f}")

        return cv_results, test_results, self.default_rf_params

    def get_cv_detailed_results(self, cv_results: dict) -> pd.DataFrame:
        """交差検証の詳細結果をDataFrameで取得"""
        detailed_results = []
        for fold in range(10):
            fold_result = {
                'Fold': fold + 1,
                'F1': cv_results['f1_scores'][fold],
                'Precision': cv_results['precision_scores'][fold],
                'Recall': cv_results['recall_scores'][fold],
                'Accuracy': cv_results['accuracy_scores'][fold],
                'ROC_AUC': cv_results['roc_auc_scores'][fold]
            }
            detailed_results.append(fold_result)

        return pd.DataFrame(detailed_results)

    def get_test_results_summary(self) -> dict:
        """テストデータ評価結果のサマリー取得"""
        if self.test_results is None:
            print("テストデータでの評価が実行されていません。")
            return {}

        return {
            'test_metrics': {
                'f1': self.test_results['f1'],
                'precision': self.test_results['precision'],
                'recall': self.test_results['recall'],
                'accuracy': self.test_results['accuracy'],
                'roc_auc': self.test_results['roc_auc']
            },
            'test_class_distribution': self.test_results['test_class_distribution'],
            'prediction_distribution': self.test_results['pred_class_distribution']
        }

    def compare_cv_and_test_results(self, cv_results: dict) -> pd.DataFrame:
        """交差検証結果とテストデータ結果の比較"""
        if self.test_results is None:
            print("テストデータでの評価が実行されていません。")
            return pd.DataFrame()

        comparison_data = []

        for metric in ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']:
            cv_mean = cv_results[f'{metric}_mean']
            cv_std = cv_results[f'{metric}_std']
            test_score = self.test_results[metric]
            difference = test_score - cv_mean

            comparison_data.append({
                '評価指標': metric.upper(),
                'CV平均': cv_mean,
                'CV標準偏差': cv_std,
                'テストスコア': test_score,
                '差分 (テスト - CV平均)': difference,
                '差分の標準偏差比': abs(difference) / cv_std if cv_std > 0 else 0
            })

        return pd.DataFrame(comparison_data)

    def get_feature_analysis(self) -> dict:
        """特徴量分析結果の取得"""
        params_to_return = self.default_rf_params.copy()

        # サンプリング情報を追加
        sampling_info = {}
        if self.original_class_distribution and self.resampled_train_distribution:
            sampling_info = {
                'original_class_0': self.original_class_distribution['class_0'],
                'original_class_1': self.original_class_distribution['class_1'],
                'original_total': self.original_class_distribution['total'],
                'resampled_train_class_0': self.resampled_train_distribution['class_0'],
                'resampled_train_class_1': self.resampled_train_distribution['class_1'],
                'resampled_train_total': self.resampled_train_distribution['total'],
                'change_rate_percent': (self.resampled_train_distribution['total'] / self.original_class_distribution['total'] - 1) * 100
            }

        return {
            'best_params': params_to_return,
            'feature_importance_scores': self.feature_importance_scores,
            'selected_features': self.selected_features,
            'all_feature_names': self.all_feature_names,
            'feature_selection_threshold': self.feature_selection_threshold,
            'tfidf_max_features': self.tfidf_max_features,
            'sampling_info': sampling_info,
            'java_tokenizer_settings': {
                'min_token_length': self.java_tokenizer.min_token_length,
                'include_package_tokens': self.java_tokenizer.include_package_tokens,
                'stopwords_count': len(self.java_tokenizer.java_stopwords)
            }
        }

    def display_feature_importance_table(self, top_n: int = 10):
        """RandomForest Feature Importanceスコアテーブルの表示"""
        if self.feature_importance_scores is None or self.selected_features is None:
            print("Feature Importanceスコアは計算されていません。")
            return

        if self.all_feature_names is None:
            print("全特徴量名が取得できません。")
            return

        # 全特徴量のFeature ImportanceスコアをDataFrameに変換
        all_features_df = pd.DataFrame({
            '特徴量': self.all_feature_names,
            'Feature Importance': self.feature_importance_scores
        }).sort_values('Feature Importance', ascending=False)

        # 選択された特徴量のみをフィルタリング
        selected_features_df = all_features_df[all_features_df['特徴量'].isin(self.selected_features)]

        # 特徴量タイプの分類
        selected_features_df['タイプ'] = selected_features_df['特徴量'].apply(
            lambda x: 'LongName TF-IDF' if x.startswith('LongName_tfidf_')
                     else 'Parent TF-IDF' if x.startswith('Parent_tfidf_')
                     else 'Project' if x.startswith('Project_')
                     else '数値'
        )

        print(f"\n=== 上位{top_n}特徴量（RandomForest Feature Importance） ===")
        display_df = selected_features_df.head(top_n)[['特徴量', 'タイプ', 'Feature Importance']].copy()
        print(display_df.to_string(index=False))

        # タイプ別の統計
        print(f"\n=== 特徴量タイプ別統計（選択された特徴量のみ） ===")
        type_stats = selected_features_df['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_score = selected_features_df[selected_features_df['タイプ'] == feature_type]['Feature Importance'].mean()
            print(f"{feature_type}: {count}個 (平均Feature Importance: {avg_score:.4f})")

        return selected_features_df

    def display_sampling_summary(self):
        """アンダーサンプリングのサマリー表示"""
        if not self.original_class_distribution or not self.resampled_train_distribution:
            print("サンプリング情報がありません。")
            return

        print("\n=== RandomUnderSampler サマリー ===")
        print(f"元データ (訓練データ):")
        print(f"  クラス 0: {self.original_class_distribution['class_0']:,}件")
        print(f"  クラス 1: {self.original_class_distribution['class_1']:,}件")
        print(f"  合計: {self.original_class_distribution['total']:,}件")

        resampled_class_0 = self.resampled_train_distribution['class_0']
        resampled_class_1 = self.resampled_train_distribution['class_1']
        resampled_total = self.resampled_train_distribution['total']

        print(f"\nアンダーサンプリング適用後 (訓練データ):")
        print(f"  クラス 0: {resampled_class_0:,}件")
        print(f"  クラス 1: {resampled_class_1:,}件")
        print(f"  合計: {resampled_total:,}件")

        change_rate = (resampled_total / self.original_class_distribution['total'] - 1) * 100
        print(f"\n元の訓練データからの変化率: {change_rate:.1f}%")

    def display_feature_selection_summary(self):
        """RandomForest Feature Importanceによる特徴量選択のサマリー表示"""
        if self.feature_importance_scores is None or self.selected_features is None:
            print("特徴量選択が実行されていません。")
            return

        print(f"\n=== RandomForest Feature Importanceによる特徴量選択サマリー ===")
        print(f"特徴量選択手法: RandomForest Feature Importance")
        print(f"選択閾値: Feature Importance ≥ {self.feature_selection_threshold}")
        print(f"元の特徴量数: {len(self.all_feature_names)}")
        print(f"選択された特徴量数: {len(self.selected_features)}")
        print(f"選択率: {len(self.selected_features)/len(self.all_feature_names)*100:.1f}%")

        # Feature Importanceの統計情報
        if self.feature_importance_scores is not None:
            print(f"Feature Importance統計:")
            print(f"  最大値: {np.max(self.feature_importance_scores):.4f}")
            print(f"  平均値: {np.mean(self.feature_importance_scores):.4f}")
            print(f"  最小値: {np.min(self.feature_importance_scores):.4f}")
            print(f"  閾値以上の特徴量数: {np.sum(self.feature_importance_scores >= self.feature_selection_threshold)}")

    def display_tokenizer_analysis(self, sample_size: int = 5):
        """Javaトークナイザーの動作例を表示"""
        print("\n=== Javaトークナイザー動作例 ===")
        print(f"設定:")
        print(f"  最小トークン長: {self.java_tokenizer.min_token_length}")
        print(f"  パッケージトークン含む: {self.java_tokenizer.include_package_tokens}")
        print(f"  ストップワード数: {len(self.java_tokenizer.java_stopwords)}")

        # サンプルデータの表示例（実際のデータがある場合）
        sample_longnames = [
            "org.elasticsearch.index.fielddata.plain.GeoPointDoubleArrayAtomicFieldData$Empty.<init>()V",
            "com.example.MyClass.calculateSum(int,int)int",
            "java.util.ArrayList.add(Object)boolean"
        ]

        sample_parents = [
            "org.elasticsearch.index.fielddata.plain.GeoPointDoubleArrayAtomicFieldData$Empty",
            "com.example.MyClass",
            "java.util.ArrayList"
        ]

        print(f"\n=== LongName トークン化例 ===")
        for i, longname in enumerate(sample_longnames[:sample_size]):
            tokens = self.java_tokenizer(longname)
            print(f"{i+1}. {longname}")
            print(f"   → {tokens}")
            if i < len(sample_longnames) - 1:
                print()

        print(f"\n=== Parent トークン化例 ===")
        for i, parent in enumerate(sample_parents[:sample_size]):
            tokens = self.java_tokenizer(parent)
            print(f"{i+1}. {parent}")
            print(f"   → {tokens}")
            if i < len(sample_parents) - 1:
                print()


# 使用例
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # BugHunter（10分割交差検証 + RandomUnderSampler + RandomForest Feature Importance使用）のインスタンス作成
    bug_hunter = BugHunter(
        feature_selection_threshold=0.001,  # Feature Importanceの閾値
        tfidf_max_features=100,
        java_tokenizer_min_length=2,
        include_package_tokens=False,
        test_size=0.2  # テストデータの比率
    )

    # パイプライン実行
    cv_results, test_results, final_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("BugHunter 10分割交差検証バグ予測完了! (RandomForest Feature Importance版)")
    print("="*60)

    # 主要な結果表示
    print(f"交差検証 平均F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
    print(f"テストデータ F1スコア: {test_results['f1']:.4f}")
    print(f"交差検証 平均ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
    print(f"テストデータ ROC-AUC: {test_results['roc_auc']:.4f}")

    # 詳細結果の表示
    detailed_df = bug_hunter.get_cv_detailed_results(cv_results)
    print(f"\n=== 各フォールドの詳細結果（交差検証） ===")
    print(detailed_df.round(4).to_string(index=False))

    # 交差検証とテストデータ結果の比較
    comparison_df = bug_hunter.compare_cv_and_test_results(cv_results)
    print(f"\n=== 交差検証 vs テストデータ結果比較 ===")
    print(comparison_df.round(4).to_string(index=False))

    # その他のサマリー表示
    bug_hunter.display_sampling_summary()
    bug_hunter.display_feature_importance_table(top_n=15)
    bug_hunter.display_feature_selection_summary()
    bug_hunter.display_tokenizer_analysis(sample_size=3)

    # 特徴量分析結果
    feature_analysis = bug_hunter.get_feature_analysis()
    print(f"\n使用パラメータ: {feature_analysis['best_params']}")
    print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}")
    print(f"全特徴量数: {len(feature_analysis['all_feature_names'])}")
    print(f"Feature Importance閾値: {feature_analysis['feature_selection_threshold']}")
    print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")

    if feature_analysis['sampling_info']:
        sampling_info = feature_analysis['sampling_info']
        print(f"データサイズ変化率: {sampling_info['change_rate_percent']:.1f}%")

    # テストデータ結果のサマリー
    test_summary = bug_hunter.get_test_results_summary()
    if test_summary:
        print(f"\n=== テストデータ評価サマリー ===")
        print(f"テストデータクラス分布:")
        print(f"  クラス 0: {test_summary['test_class_distribution']['class_0']}件")
        print(f"  クラス 1: {test_summary['test_class_distribution']['class_1']}件")
        print(f"予測結果分布:")
        print(f"  予測クラス 0: {test_summary['prediction_distribution']['predicted_class_0']}件")
        print(f"  予測クラス 1: {test_summary['prediction_distribution']['predicted_class_1']}件")

    print("\n" + "="*60)
    print("論文に基づく評価完了! (RandomForest Feature Importance版)")
    print("交差検証とテストデータの両方で評価を実施")
    print("="*60)
