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

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

class JavaCodeTokenizer:
    """Javaのメソッド名やクラス名を適切にトークン化するカスタムトークナイザー"""

    def __init__(self, min_token_length: int = 3, include_package_tokens: bool = True):
        self.min_token_length = min_token_length
        self.include_package_tokens = include_package_tokens

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
        parts = re.sub(r'(?<!^)(?=[A-Z])', ' ', text).split()
        return [part for part in parts if len(part) >= self.min_token_length]

    def _split_snake_case(self, text: str) -> list[str]:
        parts = re.split(r'[_\-]', text)
        return [part for part in parts if len(part) >= self.min_token_length]

    def _extract_method_signature_tokens(self, method_signature: str) -> list[str]:
        tokens = []

        pattern = r'^(.*?)\.([^.()]+)\(([^)]*)\)(.*)$'
        match = re.match(pattern, method_signature)

        if match:
            package_and_class = match.group(1)
            method_name = match.group(2)
            parameters = match.group(3)
            return_type = match.group(4)

            if package_and_class:
                package_parts = package_and_class.split('.')

                if self.include_package_tokens:
                    for part in package_parts[:-1]:
                        if part and len(part) >= self.min_token_length:
                            tokens.append(part.lower())

                if package_parts:
                    class_name = package_parts[-1]
                    class_parts = class_name.split('$')
                    for class_part in class_parts:
                        if class_part:
                            camel_tokens = self._split_camel_case(class_part)
                            tokens.extend([token.lower() for token in camel_tokens])

            if method_name and method_name != '<init>' and method_name != '<clinit>':
                snake_tokens = self._split_snake_case(method_name)
                for token in snake_tokens:
                    camel_tokens = self._split_camel_case(token)
                    tokens.extend([token.lower() for token in camel_tokens])
            elif method_name in ['<init>', '<clinit>']:
                tokens.append('constructor')

        else:
            parts = method_signature.split('.')
            for part in parts:
                if part and '(' not in part:
                    snake_tokens = self._split_snake_case(part)
                    for token in snake_tokens:
                        camel_tokens = self._split_camel_case(token)
                        tokens.extend([token.lower() for token in camel_tokens])

        return tokens

    def _extract_class_name_tokens(self, class_name: str) -> list[str]:
        tokens = []
        parts = class_name.split('.')

        if self.include_package_tokens:
            for part in parts[:-1]:
                if part and len(part) >= self.min_token_length:
                    tokens.append(part.lower())

        if parts:
            class_part = parts[-1]
            class_components = class_part.split('$')
            for component in class_components:
                if component:
                    camel_tokens = self._split_camel_case(component)
                    tokens.extend([token.lower() for token in camel_tokens])

        return tokens

    def tokenize_longname(self, longname: str) -> list[str]:
        if not longname or pd.isna(longname):
            return []

        tokens = self._extract_method_signature_tokens(str(longname))

        filtered_tokens = [
            token for token in tokens
            if len(token) >= self.min_token_length and token.lower() not in self.java_stopwords
        ]

        return filtered_tokens

    def tokenize_parent(self, parent: str) -> list[str]:
        if not parent or pd.isna(parent):
            return []

        tokens = self._extract_class_name_tokens(str(parent))

        filtered_tokens = [
            token for token in tokens
            if len(token) >= self.min_token_length and token.lower() not in self.java_stopwords
        ]

        return filtered_tokens

    def __call__(self, text: str) -> list[str]:
        if not text or pd.isna(text):
            return []

        text_str = str(text)

        if '(' in text_str and ')' in text_str:
            return self.tokenize_longname(text_str)
        else:
            return self.tokenize_parent(text_str)


class BugHunter:
    """BugHunterデータセットに基づくバグ予測システム（RandomForest Feature Importance版）"""

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 test_size: float = 0.2):

        self.best_model = None
        self.feature_importance = None

        self.all_feature_names = None
        self.selected_features = None
        self.feature_selection_threshold = feature_selection_threshold
        self.feature_importance_scores = None

        self.initial_X = None
        self.initial_y = None
        self.tfidf_vectorizer_longname = None
        self.tfidf_vectorizer_parent = None
        self.tfidf_max_features = tfidf_max_features
        self.scaler = None

        # operation_type One-Hotエンコーディング用
        self.operation_type_columns = None
        self.has_operation_type = False

        self.test_size = test_size
        self.test_results = None

        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        self.original_class_distribution = None
        self.resampled_train_distribution = None

        self.default_rf_params = {
            'n_estimators': 500,
            'max_depth': 30,
            'random_state': GLOBAL_SEED,
            'n_jobs': -1
        }

    def read_data(self, data_path: str, max_rows: int = 100) -> pd.DataFrame:
        print(f"\n=== 1) データ読み込み（最大{max_rows}行、欠損値はそのまま保持） ===")
        df = pd.read_csv(data_path, nrows=max_rows)
        print(f"読み込み完了: {len(df)}行, {len(df.columns)}列")

        # operation_typeカラムの存在確認
        if 'operation_type' in df.columns:
            self.has_operation_type = True
            print(f"operation_typeカラムが検出されました")

            # operation_typeの分布を表示（NaNも含む）
            operation_counts = df['operation_type'].value_counts(dropna=False)
            print(f"operation_type分布:")
            for op_type, count in operation_counts.items():
                percentage = count / len(df) * 100
                op_type_str = str(op_type) if pd.notna(op_type) else "NaN"
                print(f"  {op_type_str}: {count}個 ({percentage:.1f}%)")
        else:
            self.has_operation_type = False
            print(f"operation_typeカラムは見つかりませんでした")

        # 欠損値の状況を表示（ゼロフィルはしない）
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()

        print(f"欠損値の状況: {total_missing}個の欠損値")
        if total_missing > 0:
            print("主な欠損値を持つカラム:")
            for col, missing_count in missing_info[missing_info > 0].items():
                percentage = missing_count / len(df) * 100
                print(f"  {col}: {missing_count}個 ({percentage:.1f}%)")
            print("注意: 欠損値はそのまま保持され、RandomForestで自動的に処理されます")

        return df

    def _prepare_operation_type_features(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """operation_typeカラムのOne-Hotエンコーディングを行う"""
        if not self.has_operation_type or 'operation_type' not in data.columns:
            return pd.DataFrame(index=data.index)

        if is_training:
            # 訓練時：全カテゴリを記録してOne-Hotエンコーディング
            # NaNも'NaN'として明示的に処理
            operation_type_filled = data['operation_type'].fillna('NaN')
            operation_type_dummies = pd.get_dummies(operation_type_filled, prefix='operation_type')
            self.operation_type_columns = operation_type_dummies.columns.tolist()

            print(f"operation_type One-Hotエンコーディング:")
            print(f"  生成されたカラム: {self.operation_type_columns}")

            return operation_type_dummies
        else:
            # 予測時：訓練時と同じカラム構成を保持
            if self.operation_type_columns is None:
                raise ValueError("operation_typeのカラム情報が学習されていません。まず訓練データでprepare_dataを実行してください。")

            operation_type_filled = data['operation_type'].fillna('NaN')
            operation_type_dummies = pd.get_dummies(operation_type_filled, prefix='operation_type')

            # 訓練時と同じカラム構成に合わせる
            operation_type_df = pd.DataFrame(0, index=data.index, columns=self.operation_type_columns)

            # 存在するカラムのみ値を設定
            for col in operation_type_dummies.columns:
                if col in operation_type_df.columns:
                    operation_type_df[col] = operation_type_dummies[col]

            return operation_type_df

    def prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> tuple:
        print("\n=== データ前処理（カスタムJavaトークナイザー + TF-IDF + operation_type One-Hot + 正規化、欠損値保持）===")

        y = (data["Number of Bugs"] > 0.5).astype(int) if "Number of Bugs" in data.columns and is_training else None

        # 数値特徴量の処理（operation_typeは除外、欠損値はそのまま保持）
        numerical_feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_feature_columns = [col for col in numerical_feature_columns if col not in ["Number of Bugs"]]
        X_numerical = data[numerical_feature_columns].copy()

        # 無限値のみをNaNに変換（元のNaNは保持）
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)

        # RandomForestは欠損値を自動的に処理するため、ここでは欠損値を埋めない
        missing_count = X_numerical.isnull().sum().sum()
        if missing_count > 0:
            print(f"数値特徴量の欠損値: {missing_count}個（RandomForestで自動処理）")

        if is_training:
            self.scaler = StandardScaler()
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        else:
            if self.scaler is None:
                raise ValueError("Scalerが学習されていません。まず訓練データでprepare_dataを実行してください。")
            X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X_numerical.index)

        # operation_typeのOne-Hotエンコーディング
        X_operation_type_df = self._prepare_operation_type_features(data, is_training)

        # LongName TF-IDF処理（欠損値は空文字として処理）
        longname_data = data['LongName'].fillna("").astype(str)
        if is_training:
            self.tfidf_vectorizer_longname = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                tokenizer=self.java_tokenizer,
                lowercase=False,
                token_pattern=None
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

        # Parent TF-IDF処理（欠損値は空文字として処理）
        parent_data = data['Parent'].fillna("").astype(str)
        if is_training:
            self.tfidf_vectorizer_parent = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                tokenizer=self.java_tokenizer,
                lowercase=False,
                token_pattern=None
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

        # 全特徴量を結合
        X_parts = [X_numerical_scaled_df, X_longname_tfidf_df, X_parent_tfidf_df]
        if len(X_operation_type_df.columns) > 0:
            X_parts.append(X_operation_type_df)

        X = pd.concat(X_parts, axis=1)

        if is_training:
            self.all_feature_names = X.columns.tolist()
            self.initial_X = X
            self.initial_y = y

            print(f"初期使用特徴量数 (数値 + Java TF-IDF + operation_type): {len(X.columns)}")
            print(f"  - 数値特徴量: {len(numerical_feature_columns)}")
            print(f"  - LongName TF-IDF: {X_longname_tfidf.shape[1]}")
            print(f"  - Parent TF-IDF: {X_parent_tfidf.shape[1]}")
            if len(X_operation_type_df.columns) > 0:
                print(f"  - operation_type One-Hot: {len(X_operation_type_df.columns)}")
            else:
                print(f"  - operation_type One-Hot: 0 (カラムなし)")
            print(f"ラベル分布: 0={sum(y==0)}, 1={sum(y==1)}")

            # 最終的な特徴量行列の欠損値状況を確認
            final_missing = X.isnull().sum().sum()
            if final_missing > 0:
                print(f"最終特徴量行列の欠損値: {final_missing}個（主に数値特徴量）")

            self.original_class_distribution = {
                'class_0': sum(y==0),
                'class_1': sum(y==1),
                'total': len(y)
            }
            class_ratio = sum(y==1) / len(y)
            print(f"クラス分布: {class_ratio:.3f} (1の割合)")

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

            if self.has_operation_type:
                sample_op_type = data['operation_type'].iloc[0] if len(data) > 0 else None
                sample_op_type_str = str(sample_op_type) if pd.notna(sample_op_type) else 'NaN'
                print(f"\noperation_type例: {sample_op_type_str}")
                print(f"→ One-Hot: {X_operation_type_df.iloc[0].to_dict() if len(X_operation_type_df) > 0 else 'N/A'}")
        else:
            print(f"予測データの前処理完了: {len(X.columns)}列")

        return X, y

    def apply_undersampling(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        print(f"サンプリング前 - クラス0: {sum(y==0)}件, クラス1: {sum(y==1)}件", end=" → ")

        try:
            sampler = RandomUnderSampler(random_state=GLOBAL_SEED)
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            y_balanced = pd.Series(y_resampled)

            print(f"クラス0: {sum(y_balanced==0)}件, クラス1: {sum(y_balanced==1)}件")

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
        print(f"\n=== RandomForest Feature Importanceによる特徴量選択 ===")
        print(f"特徴量選択前の数: {len(X.columns)}")

        rf_selector = RandomForestClassifier(**self.default_rf_params)

        try:
            rf_selector.fit(X, y)
            self.feature_importance_scores = rf_selector.feature_importances_

            selected_indices = np.where(self.feature_importance_scores >= self.feature_selection_threshold)[0]

            if len(selected_indices) == 0:
                print(f"閾値 {self.feature_selection_threshold} で特徴量が選択されませんでした。")
                print("上位50%の特徴量を選択します。")
                n_features_to_select = max(1, len(X.columns) // 2)
                selected_indices = np.argsort(self.feature_importance_scores)[-n_features_to_select:]

            self.selected_features = [X.columns[i] for i in selected_indices]

            X_selected_df = X[self.selected_features].copy()

            print(f"選択された特徴量数 (重要度≥{self.feature_selection_threshold}): {len(self.selected_features)}")

            longname_count = len([f for f in self.selected_features if f.startswith('LongName_tfidf_')])
            parent_count = len([f for f in self.selected_features if f.startswith('Parent_tfidf_')])
            operation_type_count = len([f for f in self.selected_features if f.startswith('operation_type_')])
            numerical_count = len(self.selected_features) - longname_count - parent_count - operation_type_count

            print(f"  - 数値特徴量: {numerical_count}")
            print(f"  - LongName TF-IDF: {longname_count}")
            print(f"  - Parent TF-IDF: {parent_count}")
            print(f"  - operation_type One-Hot: {operation_type_count}")

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
        print(f"\n=== {k_folds}分割交差検証による評価 ===")

        rf = RandomForestClassifier(**self.default_rf_params)

        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)

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

        fold_results = {metric: [] for metric in scoring.keys()}

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"フォールド {fold_idx + 1}/{k_folds} 処理中...", end=" ")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            X_train_fold_resampled, y_train_fold_resampled = self.apply_undersampling(
                X_train_fold, y_train_fold
            )

            rf_fold = RandomForestClassifier(**self.default_rf_params)
            rf_fold.fit(X_train_fold_resampled, y_train_fold_resampled)

            y_pred = rf_fold.predict(X_val_fold)
            y_pred_proba = rf_fold.predict_proba(X_val_fold)[:, 1]

            fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_results['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            fold_results['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_results['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))

            print(f"F1: {fold_results['f1'][-1]:.3f}")

        cv_results = {}
        for metric in scoring.keys():
            scores = fold_results[metric]
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores

        return cv_results

    def evaluate_on_test_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        print("\n=== テストデータでの最終モデル評価 ===")

        if self.best_model is None:
            raise ValueError("最終モデルが学習されていません。まずtrain_final_model()を実行してください。")

        y_test_pred = self.best_model.predict(X_test)
        y_test_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        test_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }

        test_class_distribution = {
            'class_0': sum(y_test == 0),
            'class_1': sum(y_test == 1),
            'total': len(y_test)
        }

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

        self.test_results = {
            **test_results,
            'test_class_distribution': test_class_distribution,
            'pred_class_distribution': pred_class_distribution
        }

        return self.test_results

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        print("\n=== 最終RandomForestモデル学習 ===")

        X_resampled, y_resampled = self.apply_undersampling(X, y)

        self.best_model = RandomForestClassifier(**self.default_rf_params)
        self.best_model.fit(X_resampled, y_resampled)

        print("最終RandomForestモデル学習完了")
        print(f"パラメータ: {self.default_rf_params}")
        print(f"学習データ: {len(X_resampled)}件")

        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
            print("特徴量重要度を取得しました。")

            if len(self.feature_importance) > 0 and len(self.selected_features) == len(self.feature_importance):
                top_indices = self.feature_importance.argsort()[-5:][::-1]
                print("最終モデル特徴量重要度 上位5:")
                for i, idx in enumerate(top_indices):
                    feature_name = self.selected_features[idx]
                    importance = self.feature_importance[idx]
                    print(f"  {i+1}. {feature_name}: {importance:.4f}")

        return self.best_model

    def predict(self, X: pd.DataFrame) -> tuple:
        if self.best_model is None:
            raise ValueError("モデルが学習されていません。まずrun_pipeline()を実行してください。")
        if self.selected_features is None:
            raise ValueError("特徴量選択が実行されていません。まずrun_pipeline()を実行してください。")
        if self.tfidf_vectorizer_longname is None or self.tfidf_vectorizer_parent is None:
            raise ValueError("TF-IDF Vectorizerが学習されていません。まずrun_pipeline()を実行してください。")
        if self.scaler is None:
            raise ValueError("Scalerが学習されていません。まずrun_pipeline()を実行してください。")

        # 数値特徴量の処理（欠損値はそのまま保持）
        numerical_feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if "Number of Bugs" in numerical_feature_columns:
            numerical_feature_columns.remove("Number of Bugs")

        X_numerical = X[numerical_feature_columns].copy()
        # 無限値のみをNaNに変換（元のNaNは保持）
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)

        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X_numerical.index)

        # operation_typeの処理
        X_operation_type_df = self._prepare_operation_type_features(X, is_training=False)

        # LongName TF-IDF処理
        longname_data_pred = X['LongName'].fillna("").astype(str)
        X_longname_tfidf_pred = self.tfidf_vectorizer_longname.transform(longname_data_pred)
        X_longname_tfidf_df_pred = pd.DataFrame(X_longname_tfidf_pred.toarray(),
                                                  columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf_pred.shape[1])],
                                                  index=X.index)

        # Parent TF-IDF処理
        parent_data_pred = X['Parent'].fillna("").astype(str)
        X_parent_tfidf_pred = self.tfidf_vectorizer_parent.transform(parent_data_pred)
        X_parent_tfidf_df_pred = pd.DataFrame(X_parent_tfidf_pred.toarray(),
                                                columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf_pred.shape[1])],
                                                index=X.index)

        # 全特徴量を結合
        X_parts = [X_numerical_scaled_df, X_longname_tfidf_df_pred, X_parent_tfidf_df_pred]
        if len(X_operation_type_df.columns) > 0:
            X_parts.append(X_operation_type_df)

        X_processed_full = pd.concat(X_parts, axis=1)

        # 選択された特徴量のみを使用
        X_processed_df = X_processed_full[self.selected_features]

        y_pred_proba = self.best_model.predict_proba(X_processed_df)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred, y_pred_proba

    def run_pipeline(self, data_path: str, max_rows: int = 100):
        print("=== BugHunter 10分割交差検証バグ予測パイプライン (RandomForest Feature Importance版) ===")
        print(f"- データ制限: 最大{max_rows}行、欠損値保持（RandomForestで自動処理）")

        data = self.read_data(data_path, max_rows)

        X_full, y_full = self.prepare_data(data, is_training=True)

        print(f"\n=== 訓練・テストデータ分割 (テスト比率: {self.test_size}) ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=self.test_size,
            random_state=GLOBAL_SEED,
            stratify=y_full
        )

        print(f"訓練データ: {len(X_train)}件 (クラス0: {sum(y_train==0)}, クラス1: {sum(y_train==1)})")
        print(f"テストデータ: {len(X_test)}件 (クラス0: {sum(y_test==0)}, クラス1: {sum(y_test==1)})")

        print("\n=== 特徴量選択器の学習（訓練データのみ使用） ===")
        X_train_temp_resampled, y_train_temp_resampled = self.apply_undersampling(X_train, y_train)
        X_reduced = self.select_features_by_rf_importance(X_train_temp_resampled, y_train_temp_resampled)

        X_train_reduced_df = X_train[self.selected_features]
        X_test_reduced_df = X_test[self.selected_features]

        print("\n=== 交差検証評価（訓練データのみ使用） ===")
        cv_results = self.evaluate_model_with_cv(X_train_reduced_df, y_train, k_folds=10)

        final_model = self.train_final_model(X_train_reduced_df, y_train)

        test_results = self.evaluate_on_test_data(X_test_reduced_df, y_test)

        print("\n=== 10分割交差検証結果（訓練データ） ===")
        print(f"F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        print(f"Recall: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        print(f"Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")

        return cv_results, test_results, self.default_rf_params

    def get_cv_detailed_results(self, cv_results: dict) -> pd.DataFrame:
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
        params_to_return = self.default_rf_params.copy()

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
            },
            'operation_type_info': {
                'has_operation_type': self.has_operation_type,
                'operation_type_columns': self.operation_type_columns
            }
        }

    def display_feature_importance_table(self, top_n: int = 10):
        if self.feature_importance_scores is None or self.selected_features is None:
            print("Feature Importanceスコアは計算されていません。")
            return

        if self.all_feature_names is None:
            print("全特徴量名が取得できません。")
            return

        all_features_df = pd.DataFrame({
            '特徴量': self.all_feature_names,
            'Feature Importance': self.feature_importance_scores
        }).sort_values('Feature Importance', ascending=False)

        selected_features_df = all_features_df[all_features_df['特徴量'].isin(self.selected_features)]

        selected_features_df['タイプ'] = selected_features_df['特徴量'].apply(
            lambda x: 'LongName TF-IDF' if x.startswith('LongName_tfidf_')
                      else 'Parent TF-IDF' if x.startswith('Parent_tfidf_')
                      else 'operation_type' if x.startswith('operation_type_')
                      else '数値'
        )

        print(f"\n=== 上位{top_n}特徴量（RandomForest Feature Importance） ===")
        display_df = selected_features_df.head(top_n)[['特徴量', 'タイプ', 'Feature Importance']].copy()
        print(display_df.to_string(index=False))

        print(f"\n=== 特徴量タイプ別統計（選択された特徴量のみ） ===")
        type_stats = selected_features_df['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_score = selected_features_df[selected_features_df['タイプ'] == feature_type]['Feature Importance'].mean()
            print(f"{feature_type}: {count}個 (平均Feature Importance: {avg_score:.4f})")

        return selected_features_df

    def display_sampling_summary(self):
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
        if self.feature_importance_scores is None or self.selected_features is None:
            print("特徴量選択が実行されていません。")
            return

        print(f"\n=== RandomForest Feature Importanceによる特徴量選択サマリー ===")
        print(f"特徴量選択手法: RandomForest Feature Importance")
        print(f"選択閾値: Feature Importance ≥ {self.feature_selection_threshold}")
        print(f"元の特徴量数: {len(self.all_feature_names)}")
        print(f"選択された特徴量数: {len(self.selected_features)}")
        print(f"選択率: {len(self.selected_features)/len(self.all_feature_names)*100:.1f}%")

        if self.feature_importance_scores is not None:
            print(f"Feature Importance統計:")
            print(f"  最大値: {np.max(self.feature_importance_scores):.4f}")
            print(f"  平均値: {np.mean(self.feature_importance_scores):.4f}")
            print(f"  最小値: {np.min(self.feature_importance_scores):.4f}")
            print(f"  閾値以上の特徴量数: {np.sum(self.feature_importance_scores >= self.feature_selection_threshold)}")

    def display_operation_type_analysis(self):
        """operation_type特徴量の分析結果を表示"""
        if not self.has_operation_type:
            print("\noperation_typeカラムは存在しません。")
            return

        print(f"\n=== operation_type One-Hotエンコーディング分析 ===")
        if self.operation_type_columns:
            print(f"生成されたoperation_typeカラム:")
            for i, col in enumerate(self.operation_type_columns):
                print(f"  {i+1}. {col}")

            # 選択された特徴量の中でoperation_type関連のものをチェック
            if self.selected_features:
                selected_op_features = [f for f in self.selected_features if f.startswith('operation_type_')]
                print(f"\n選択されたoperation_type特徴量: {len(selected_op_features)}個")
                for feature in selected_op_features:
                    if self.feature_importance_scores is not None and self.all_feature_names:
                        try:
                            idx = self.all_feature_names.index(feature)
                            importance = self.feature_importance_scores[idx]
                            print(f"  {feature}: {importance:.4f}")
                        except (ValueError, IndexError):
                            print(f"  {feature}: 重要度不明")
                    else:
                        print(f"  {feature}")
        else:
            print("operation_typeカラム情報が取得できませんでした。")

    def display_tokenizer_analysis(self, sample_size: int = 5):
        print("\n=== Javaトークナイザー動作例 ===")
        print(f"設定:")
        print(f"  最小トークン長: {self.java_tokenizer.min_token_length}")
        print(f"  パッケージトークン含む: {self.java_tokenizer.include_package_tokens}")
        print(f"  ストップワード数: {len(self.java_tokenizer.java_stopwords)}")

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

    def run_simple_pipeline(self, csv_path: str, max_rows: int = 100):
        """元のrun_pipelineのエイリアス"""
        return self.run_pipeline(csv_path, max_rows)

    def display_results(self, cv_results: dict, test_results: dict):
        """結果表示（元のコードとの互換性のため）"""
        print("\n" + "="*50)
        print("=== BugHunter 結果サマリー ===")
        print("="*50)

        print(f"\n=== 交差検証結果 ===")
        print(f"F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        print(f"Recall: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        print(f"Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")

        print(f"\n=== テストデータ結果 ===")
        print(f"F1スコア: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"ROC-AUC: {test_results['roc_auc']:.4f}")

        if self.selected_features and self.feature_importance is not None:
            print(f"\n=== 特徴量情報 ===")
            print(f"選択された特徴量数: {len(self.selected_features)}")
            print(f"上位3重要特徴量:")
            top_indices = np.argsort(self.feature_importance)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                feature_name = self.selected_features[idx]
                importance = self.feature_importance[idx]
                print(f"  {i+1}. {feature_name}: {importance:.4f}")

def main():
    """使用例"""
    # BugHunterインスタンス作成
    bug_hunter = BugHunter(
        feature_selection_threshold=0.001,    # Feature Importance閾値
        tfidf_max_features=100,               # TF-IDF最大特徴量数
        java_tokenizer_min_length=2,          # トークンの最小長
        include_package_tokens=False,         # パッケージトークンを含むか
        test_size=0.2                        # テストデータの比率
    )

    print("BugHunterインスタンス作成完了")
    print(f"設定:")
    print(f"  Feature Importance閾値: {bug_hunter.feature_selection_threshold}")
    print(f"  TF-IDF最大特徴量数: {bug_hunter.tfidf_max_features}")
    print(f"  Javaトークナイザー最小長: {bug_hunter.java_tokenizer.min_token_length}")
    print(f"  パッケージトークン含む: {bug_hunter.java_tokenizer.include_package_tokens}")
    print(f"  テストサイズ: {bug_hunter.test_size}")

    try:
        # データファイルのパスを指定
        data_path = "method-p_filtered_v2_changes_nan.csv"

        # パイプライン実行
        cv_results, test_results, final_params = bug_hunter.run_pipeline(
            data_path,
            max_rows=3000
        )

        print("="*60)
        print("BugHunter 10分割交差検証バグ予測完了!")
        print("="*60)

        # 詳細分析
        print(f"交差検証 平均F1スコア: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"テストデータ F1スコア: {test_results['f1']:.4f}")
        print(f"交差検証 平均ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        print(f"テストデータ ROC-AUC: {test_results['roc_auc']:.4f}")

        # 各フォールドの詳細結果
        detailed_df = bug_hunter.get_cv_detailed_results(cv_results)
        print(f"\n=== 各フォールドの詳細結果（交差検証） ===")
        print(detailed_df.round(4))

        # 交差検証とテストデータの結果比較
        comparison_df = bug_hunter.compare_cv_and_test_results(cv_results)
        print(f"\n=== 交差検証 vs テストデータ結果比較 ===")
        print(comparison_df.round(4))

        # 特徴量の詳細分析
        bug_hunter.display_sampling_summary()
        feature_importance_df = bug_hunter.display_feature_importance_table(top_n=15)
        bug_hunter.display_feature_selection_summary()
        bug_hunter.display_operation_type_analysis()  # 新しい分析メソッド
        bug_hunter.display_tokenizer_analysis(sample_size=3)

        # 特徴量分析の詳細取得
        feature_analysis = bug_hunter.get_feature_analysis()
        print(f"\n=== パラメータ・設定サマリー ===")
        print(f"使用パラメータ: {feature_analysis['best_params']}")
        print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}")
        print(f"全特徴量数: {len(feature_analysis['all_feature_names'])}")
        print(f"Feature Importance閾値: {feature_analysis['feature_selection_threshold']}")
        print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")
        print(f"operation_type使用: {feature_analysis['operation_type_info']['has_operation_type']}")

        if feature_analysis['sampling_info']:
            sampling_info = feature_analysis['sampling_info']
            print(f"データサイズ変化率: {sampling_info['change_rate_percent']:.1f}%")

    except FileNotFoundError:
        print(f"エラー: ファイル '{data_path}' が見つかりません。")
        print("正しいファイルパスを指定してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
