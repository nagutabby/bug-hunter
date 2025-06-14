import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler  # SMOTEからRandomUnderSamplerに変更
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


class BaseBugHunter:
    """
    バグ予測の基底クラス
    RandomForestとk-NNで共通の機能を提供
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 100,
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

        # Javaトークナイザーの初期化
        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        # サンプリング関連の属性
        self.sampler = None
        self.resampled_X = None
        self.resampled_y = None
        self.original_class_distribution = None
        self.resampled_train_distribution = None

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
        print(f"\n=== RandomUnderSampler アンダーサンプリング適用 ===")
        print(f"サンプリング前 - クラス0: {sum(y==0)}件, クラス1: {sum(y==1)}件")

        try:
            self.sampler = RandomUnderSampler(random_state=GLOBAL_SEED)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)

            X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            y_balanced = pd.Series(y_resampled)

            print(f"サンプリング後 - クラス0: {sum(y_balanced==0)}件, クラス1: {sum(y_balanced==1)}件")
            print(f"総データ数: {len(X_balanced)}件 (減少率: {(1 - len(X_balanced) / len(X)) * 100:.1f}%)")

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

    def log_loss_function(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Log Loss損失関数"""
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred_proba_clipped)

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

    def comprehensive_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """包括的評価（デフォルトしきい値0.5のみ使用）"""
        print("\n=== 包括的評価 ===")

        # 予測確率取得
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # デフォルトしきい値（0.5）での予測
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 評価指標の計算
        results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'LogLoss': self.log_loss_function(y_test, y_pred_proba),
            'ROC_AUC': roc_auc_score(y_test, y_pred_proba),
            'Threshold': 0.5
        }

        print("\n--- 評価結果（しきい値=0.5）---")
        for metric, score in results.items():
            if metric != 'Threshold':
                print(f"{metric}: {score:.4f}")

        return results, y_pred, y_pred_proba

    def predict(self, X: pd.DataFrame) -> tuple:
        """予測（基底クラス用の共通実装）"""
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
        X_processed = pd.DataFrame(0, index=X_processed_full.index, columns=self.selected_features)
        for col in self.selected_features:
            if col in X_processed_full.columns:
                X_processed[col] = X_processed_full[col]

        # 予測確率取得
        y_pred_proba = self.best_model.predict_proba(X_processed)[:, 1]

        # デフォルトしきい値での予測
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred, y_pred_proba

    def get_feature_analysis(self) -> dict:
        """特徴量分析結果の取得"""
        params_to_return = self.best_params.copy() if self.best_params else {}

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
            'optimization_history': self.optimization_history,
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'all_feature_names': self.all_feature_names,
            'feature_selection_threshold': self.feature_selection_threshold,
            'tfidf_max_features': self.tfidf_max_features,
            'sampling_info': sampling_info, # 'downsampling_info'から'sampling_info'に変更
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
                print("エラー: 選択された特徴量名と重要度の数が一致しません。")
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

        print(f"\n=== 上位{top_n}特徴量重要度 ===")
        display_df = importance_df_filtered.head(top_n)[['特徴量', 'タイプ', '重要度']].copy()
        print(display_df.to_string(index=False))

        # タイプ別の統計
        print(f"\n=== 特徴量タイプ別統計 ===")
        type_stats = importance_df_filtered['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_importance = importance_df_filtered[importance_df_filtered['タイプ'] == feature_type]['重要度'].mean()
            print(f"{feature_type}: {count}個 (平均重要度: {avg_importance:.4f})")

        return importance_df_filtered

    def display_sampling_summary(self):
        """アンダーサンプリングのサマリー表示"""
        if not self.original_class_distribution or not self.resampled_train_distribution:
            print("サンプリング情報がありません。")
            return

        print("\n=== RandomUnderSampler サマリー ===")
        print(f"元データ (全データ):")
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
        print(f"\n元データ全体からの変化率: {change_rate:.1f}%")

        original_imbalance = self.original_class_distribution['class_0'] / self.original_class_distribution['class_1'] if self.original_class_distribution['class_1'] > 0 else float('inf')
        new_imbalance = resampled_class_0 / resampled_class_1 if resampled_class_1 > 0 else float('inf')
        print(f"クラス不均衡比 (クラス0/クラス1):")
        print(f"  元データ (全データ): {original_imbalance:.2f}:1")
        print(f"  適用後 (訓練データ): {new_imbalance:.2f}:1")

        # 削減されたサンプル数
        removed_samples = self.original_class_distribution['class_0'] - resampled_class_0
        print(f"\n削減された多数派クラスのサンプル数: {removed_samples:,}件")

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

        # サンプルの取得と表示
        if hasattr(self, 'initial_X'):
            # 元データから LongName と Parent を取得
            data_sample = pd.read_csv("method-p.csv").head(sample_size)

            print(f"\n=== LongName トークン化例 (上位{sample_size}件) ===")
            for i, longname in enumerate(data_sample['LongName'].head(sample_size)):
                tokens = self.java_tokenizer(str(longname))
                print(f"{i+1}. {longname}")
                print(f"   → {tokens}")
                if i < sample_size - 1:
                    print()

            print(f"\n=== Parent トークン化例 (上位{sample_size}件) ===")
            for i, parent in enumerate(data_sample['Parent'].head(sample_size)):
                tokens = self.java_tokenizer(str(parent))
                print(f"{i+1}. {parent}")
                print(f"   → {tokens}")
                if i < sample_size - 1:
                    print()

    # 抽象メソッド（サブクラスで実装）
    def evaluate_model_with_cv(self, params: dict, X: pd.DataFrame, y: pd.Series, k_folds: int = 3) -> float:
        """交差検証を用いたモデル評価（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装してください")

    def optimize_hyperparameters_with_log_loss(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 15) -> dict:
        """ハイパーパラメータ最適化（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装してください")

    def train_initial_model_for_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """特徴量重要度取得のための初期モデル学習（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装してください")

    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, optimal_params: dict):
        """最適化されたパラメータでモデルを学習（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装してください")

    def run_pipeline(self, data_path: str):
        """パイプラインの実行（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装してください")
