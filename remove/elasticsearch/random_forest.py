import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語フォント対応
import re
from typing import Set
import warnings
import pickle
import os

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

        # 予測結果保存用（新機能）
        self.predictions_data = None
        self.test_indices = None

        # PDP用のデータ保存
        self.X_train_for_pdp = None
        self.y_train_for_pdp = None

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

        # 予測結果を保存（新機能）
        self.predictions_data = {
            'y_true': y_test.values,
            'y_pred': y_test_pred,
            'y_pred_proba': y_test_pred_proba,
            'test_indices': self.test_indices
        }

        self.test_results = {
            **test_results,
            'test_class_distribution': test_class_distribution,
            'pred_class_distribution': pred_class_distribution
        }

        return self.test_results

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        print("\n=== 最終RandomForestモデル学習 ===")

        X_resampled, y_resampled = self.apply_undersampling(X, y)

        # PDP用にデータを保存
        self.X_train_for_pdp = X_resampled.copy()
        self.y_train_for_pdp = y_resampled.copy()

        self.best_model = RandomForestClassifier(**self.default_rf_params)
        self.best_model.fit(X_resampled, y_resampled)

        print("最終RandomForestモデル学習完了")
        print(f"パラメータ: {self.default_rf_params}")
        print(f"学習データ: {len(X_resampled)}件")

        return self.best_model

    def plot_partial_dependence(self, top_n: int = 20, save_path: str = None):
        """特徴量上位N個のPartial Dependence Plotを描画"""

        if self.best_model is None:
            raise ValueError("モデルが学習されていません。まずrun_pipeline()を実行してください。")

        if self.X_train_for_pdp is None or self.feature_importance_scores is None:
            raise ValueError("PDP用のデータが準備されていません。まずrun_pipeline()を実行してください。")

        if self.selected_features is None:
            raise ValueError("特徴量選択が実行されていません。まずrun_pipeline()を実行してください。")

        print(f"\n=== 特徴量上位{top_n}個のPartial Dependence Plot描画 ===")

        # 選択された特徴量の中で重要度上位N個を取得
        selected_features_df = pd.DataFrame({
            '特徴量': self.all_feature_names,
            'Feature Importance': self.feature_importance_scores
        })

        # 選択された特徴量のみをフィルタ
        selected_features_df = selected_features_df[
            selected_features_df['特徴量'].isin(self.selected_features)
        ].sort_values('Feature Importance', ascending=False)

        top_features = selected_features_df.head(top_n)['特徴量'].tolist()

        print(f"PDP対象特徴量（上位{len(top_features)}個）:")
        for i, feature in enumerate(top_features, 1):
            importance_idx = self.all_feature_names.index(feature)
            importance = self.feature_importance_scores[importance_idx]
            feature_type = self._get_feature_type(feature)
            unique_values = len(self.X_train_for_pdp[feature].unique())
            print(f"  {i:2d}. {feature} ({feature_type}) - 重要度: {importance:.4f} (ユニーク値: {unique_values})")

        # PDP用の特徴量インデックスを取得
        feature_indices = [self.selected_features.index(feature) for feature in top_features]

        try:
            # seaborn と matplotlib の設定
            sns.set_style("whitegrid")
            sns.set_palette("husl")

            # seabornのフォント設定で日本語フォントを指定
            sns.set(font='IPAexGothic')

            # または、seaborn設定後に japanize を再実行
            # japanize_matplotlib.japanize()

            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['figure.titlesize'] = 16

            # グリッドサイズを計算（5列固定）
            n_cols = 5
            n_rows = (len(top_features) + n_cols - 1) // n_cols

            # 図のサイズを調整
            fig_width = n_cols * 4
            fig_height = n_rows * 3

            print(f"PDP描画中... ({n_rows}行 × {n_cols}列のグリッド)")

            # Partial Dependence Plotを描画
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

            # axesを1次元配列に変換（単一行の場合の対応）
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)

            # PDPを一つずつ描画
            for i, (feature_idx, feature_name) in enumerate(zip(feature_indices, top_features)):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]

                try:
                    # 特徴量のユニーク値数をチェック
                    feature_data = self.X_train_for_pdp[feature_name]
                    unique_values = feature_data.unique()
                    n_unique = len(unique_values)

                    # 二値特徴量（0,1のみ）やカテゴリ変数の場合は棒グラフで表示
                    if n_unique <= 2 or feature_name.startswith('operation_type_'):
                        self._plot_binary_feature_effect(ax, feature_name, feature_data)

                    # 離散値が少ない場合（3-10個）も棒グラフ
                    elif n_unique <= 10 and all(isinstance(x, (int, np.integer)) for x in unique_values if not pd.isna(x)):
                        self._plot_discrete_feature_effect(ax, feature_name, feature_data)

                    # 連続値の場合は通常のPDP
                    else:
                        display = PartialDependenceDisplay.from_estimator(
                            self.best_model,
                            self.X_train_for_pdp,
                            features=[feature_idx],
                            ax=ax,
                            random_state=GLOBAL_SEED,
                            grid_resolution=min(50, max(10, n_unique))  # グリッド解像度を調整
                        )

                    # タイトルを設定（特徴量名を短縮）
                    short_name = self._shorten_feature_name(feature_name)
                    importance_idx = self.all_feature_names.index(feature_name)
                    importance = self.feature_importance_scores[importance_idx]
                    ax.set_title(f'{short_name}\n(重要度: {importance:.3f})', fontsize=10, pad=10)

                    # 軸ラベルのフォントサイズを調整
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    if not ax.get_xlabel():
                        ax.set_xlabel('特徴量の値', fontsize=9)
                    if not ax.get_ylabel():
                        ax.set_ylabel('バグ発生への影響', fontsize=9)

                    # 軸の目盛りを小数点第3位まで表示
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

                except Exception as e:
                    print(f"特徴量 '{feature_name}' のPDP描画でエラー: {e}")
                    # エラーの場合は代替表示
                    self._plot_error_alternative(ax, feature_name, e)

            # 余った subplot を非表示にする
            for i in range(len(top_features), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)

            # レイアウトを調整
            plt.tight_layout(pad=2.0)

            # メインタイトルを非表示（コメントアウト）
            # fig.suptitle(f'部分依存プロット - 特徴量重要度上位{len(top_features)}個',
            #             fontsize=16, y=0.98)

            # 保存
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                print(f"PDPを '{save_path}' に保存しました")

            plt.show()

            print("部分依存プロット描画完了")

        except Exception as e:
            print(f"PDP描画中にエラーが発生しました: {e}")
            print("代替手法: 個別特徴量のヒストグラムを表示します")
            self._plot_feature_distributions(top_features)

    def _plot_binary_feature_effect(self, ax, feature_name: str, feature_data: pd.Series):
        """二値特徴量の効果を棒グラフで表示"""
        unique_values = sorted(feature_data.unique())
        effects = []
        counts = []

        # 各値でのバグ発生率を計算
        for value in unique_values:
            mask = feature_data == value
            count = mask.sum()
            counts.append(count)
            if count > 0:
                y_subset = self.y_train_for_pdp[mask]
                bug_rate = y_subset.mean()
                effects.append(bug_rate)
            else:
                effects.append(0)

        # 棒グラフで表示（色を改善）
        colors = ['#ff6b6b' if e > 0.5 else '#4ecdc4' for e in effects]
        bars = ax.bar([str(int(v)) for v in unique_values], effects,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_ylabel('バグ発生率', fontsize=9)
        ax.set_xlabel(feature_name.replace('operation_type_', ''), fontsize=9)
        ax.set_ylim(0, max(1, max(effects) * 1.1))

        # グリッドを追加
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 数値ラベルを追加
        for bar, effect, count in zip(bars, effects, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(effects) * 0.02,
                   f'{effect:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    def _plot_discrete_feature_effect(self, ax, feature_name: str, feature_data: pd.Series):
        """離散値特徴量の効果を棒グラフで表示"""
        unique_values = sorted([x for x in feature_data.unique() if not pd.isna(x)])
        effects = []
        counts = []

        # 各値でのバグ発生率を計算
        for value in unique_values:
            mask = feature_data == value
            count = mask.sum()
            if count > 0:
                y_subset = self.y_train_for_pdp[mask]
                bug_rate = y_subset.mean()
                effects.append(bug_rate)
                counts.append(count)
            else:
                effects.append(0)
                counts.append(0)

        # カラーマップを使用（viridis）
        if max(counts) > 0:
            normalized_counts = [c/max(counts) for c in counts]
            colors = plt.cm.viridis(normalized_counts)
        else:
            colors = ['#cccccc'] * len(unique_values)

        bars = ax.bar([str(v) for v in unique_values], effects,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_ylabel('バグ発生率', fontsize=9)
        ax.set_xlabel(feature_name, fontsize=9)
        ax.set_ylim(0, max(1, max(effects) * 1.1) if effects else 1)

        # グリッドを追加
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 数値ラベルを追加
        for bar, effect, count in zip(bars, effects, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(effects) if effects else 0.1) * 0.02,
                   f'{effect:.2f}\n(n={count})', ha='center', va='bottom', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    def _plot_error_alternative(self, ax, feature_name: str, error):
        """エラーが発生した場合の代替表示"""
        ax.text(0.5, 0.5, f'描画エラー:\n{self._shorten_feature_name(feature_name)}\n\n{str(error)[:50]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8, edgecolor='red'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # 背景色を設定
        ax.set_facecolor('#f8f8f8')

    def _get_feature_type(self, feature_name: str) -> str:
        """特徴量の種類を判定"""
        if feature_name.startswith('LongName_tfidf_'):
            return 'LongName TF-IDF'
        elif feature_name.startswith('Parent_tfidf_'):
            return 'Parent TF-IDF'
        elif feature_name.startswith('operation_type_'):
            return 'operation_type'
        else:
            return '数値'

    def _shorten_feature_name(self, feature_name: str, max_length: int = 20) -> str:
        """特徴量名を短縮"""
        if len(feature_name) <= max_length:
            return feature_name

        # TF-IDF特徴量の場合は番号のみ表示
        if feature_name.startswith('LongName_tfidf_'):
            return f"LN_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('Parent_tfidf_'):
            return f"P_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('operation_type_'):
            op_type = feature_name.replace('operation_type_', '')
            return f"op_{op_type[:10]}"
        else:
            # 数値特徴量は前後を切り詰め
            if len(feature_name) > max_length:
                return feature_name[:max_length-3] + "..."
            return feature_name

    def _plot_feature_distributions(self, features: list):
        """PDPが失敗した場合の代替：特徴量分布のヒストグラム"""
        print("\n代替分析: 特徴量分布を表示")

        # matplotlib設定
        plt.rcParams['font.size'] = 10

        n_cols = 4
        n_rows = (len(features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            feature_data = self.X_train_for_pdp[feature]

            # ヒストグラムを描画（色とスタイルを改善）
            ax.hist(feature_data, bins=30, alpha=0.7, color='#4ecdc4',
                   edgecolor='black', linewidth=0.5)
            ax.set_title(self._shorten_feature_name(feature), fontsize=10)
            ax.set_xlabel('値', fontsize=9)
            ax.set_ylabel('頻度', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)

            # 軸の目盛りを小数点第3位まで表示
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            ax.grid(True, alpha=0.3, linestyle='--')

        # 余った subplot を非表示
        for i in range(len(features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        # メインタイトルを非表示（コメントアウト）
        # plt.suptitle('特徴量分布（代替表示）', fontsize=14, y=0.98)
        plt.show()

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

        # テストデータのインデックスを保存（新機能）
        self.test_indices = X_test.index.tolist()

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

    def save_predictions(self, file_path: str):
        """予測結果をファイルに保存する（新機能）"""
        if self.predictions_data is None:
            raise ValueError("予測データがありません。まずrun_pipeline()を実行してください。")

        with open(file_path, 'wb') as f:
            pickle.dump(self.predictions_data, f)
        print(f"予測結果を '{file_path}' に保存しました")

    def load_predictions(self, file_path: str):
        """予測結果をファイルから読み込む（新機能）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイル '{file_path}' が見つかりません")

        with open(file_path, 'rb') as f:
            self.predictions_data = pickle.load(f)
        print(f"予測結果を '{file_path}' から読み込みました")

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

        # 予測結果の保存（新機能）
        bug_hunter.save_predictions("predictions_nan.pkl")

        # 各フォールドの詳細結果
        detailed_df = bug_hunter.get_cv_detailed_results(cv_results)
        print(f"\n=== 各フォールドの詳細結果（交差検証） ===")
        print(detailed_df.round(4))

        # 特徴量の詳細分析
        bug_hunter.display_sampling_summary()
        feature_importance_df = bug_hunter.display_feature_importance_table(top_n=20)
        bug_hunter.display_feature_selection_summary()
        bug_hunter.display_operation_type_analysis()

        # ★ 新機能: Partial Dependence Plot描画
        print("\n" + "="*60)
        print("Partial Dependence Plot描画を開始します...")
        print("="*60)

        bug_hunter.plot_partial_dependence(
            top_n=20,
            save_path="partial_dependence_plots.png"
        )

    except FileNotFoundError:
        print(f"エラー: ファイル '{data_path}' が見つかりません。")
        print("正しいファイルパスを指定してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
