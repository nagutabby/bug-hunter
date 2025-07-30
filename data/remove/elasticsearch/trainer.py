import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
import re
from typing import Set
import warnings

warnings.filterwarnings('ignore')

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

class JavaCodeTokenizer:
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


class BugHunterTrainer:
    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 test_size: float = 0.2):

        self.feature_selection_threshold = feature_selection_threshold
        self.tfidf_max_features = tfidf_max_features
        self.test_size = test_size

        self.best_model = None
        self.tfidf_vectorizer_longname = None
        self.tfidf_vectorizer_parent = None
        self.scaler = None
        self.operation_type_columns = None

        self.all_feature_names = None
        self.selected_features = None
        self.feature_importance_scores = None

        self.original_class_distribution = None
        self.resampled_train_distribution = None
        self.has_operation_type = False

        self.java_tokenizer = JavaCodeTokenizer(
            min_token_length=java_tokenizer_min_length,
            include_package_tokens=include_package_tokens
        )

        self.default_rf_params = {
            'n_estimators': 500,
            'max_depth': 30,
            'random_state': GLOBAL_SEED,
            'n_jobs': -1
        }

        self.cv_results = None
        self.test_results = None
        self.test_indices = None
        self.predictions_data = None

    def read_data(self, data_path: str, max_rows: int = 100) -> pd.DataFrame:
        print(f"\n=== 1) データ読み込み（最大{max_rows}行、欠損値はそのまま保持） ===")
        df = pd.read_csv(data_path, nrows=max_rows)
        print(f"読み込み完了: {len(df)}行, {len(df.columns)}列")

        if 'operation_type' in df.columns:
            self.has_operation_type = True
            print(f"operation_typeカラムが検出されました")

            operation_counts = df['operation_type'].value_counts(dropna=False)
            print(f"operation_type分布:")
            for op_type, count in operation_counts.items():
                percentage = count / len(df) * 100
                op_type_str = str(op_type) if pd.notna(op_type) else "NaN"
                print(f"  {op_type_str}: {count}個 ({percentage:.1f}%)")
        else:
            self.has_operation_type = False
            print(f"operation_typeカラムは見つかりませんでした")

        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()
        print(f"欠損値の状況: {total_missing}個の欠損値")
        if total_missing > 0:
            print("主な欠損値を持つカラム:")
            for col, missing_count in missing_info[missing_info > 0].items():
                percentage = missing_count / len(df) * 100
                print(f"  {col}: {missing_count}個 ({percentage:.1f}%)")

        return df

    def _prepare_operation_type_features(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if not self.has_operation_type or 'operation_type' not in data.columns:
            return pd.DataFrame(index=data.index)

        if is_training:
            operation_type_filled = data['operation_type'].fillna('NaN')
            operation_type_dummies = pd.get_dummies(operation_type_filled, prefix='operation_type')
            self.operation_type_columns = operation_type_dummies.columns.tolist()
            print(f"operation_type One-Hotエンコーディング:")
            print(f"  生成されたカラム: {self.operation_type_columns}")
            return operation_type_dummies
        else:
            if self.operation_type_columns is None:
                raise ValueError("operation_typeのカラム情報が学習されていません。")

            operation_type_filled = data['operation_type'].fillna('NaN')
            operation_type_dummies = pd.get_dummies(operation_type_filled, prefix='operation_type')

            operation_type_df = pd.DataFrame(0, index=data.index, columns=self.operation_type_columns)

            for col in operation_type_dummies.columns:
                if col in operation_type_df.columns:
                    operation_type_df[col] = operation_type_dummies[col]

            return operation_type_df

    def prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> tuple:
        print("\n=== データ前処理（カスタムJavaトークナイザー + TF-IDF + operation_type One-Hot + 正規化）===")

        y = (data["Number of Bugs"] > 0.5).astype(int) if "Number of Bugs" in data.columns and is_training else None

        numerical_feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_feature_columns = [col for col in numerical_feature_columns if col not in ["Number of Bugs"]]
        X_numerical = data[numerical_feature_columns].copy()
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)

        missing_count = X_numerical.isnull().sum().sum()
        if missing_count > 0:
            print(f"数値特徴量の欠損値: {missing_count}個（RandomForestで自動処理）")

        if is_training:
            self.scaler = StandardScaler()
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        else:
            if self.scaler is None:
                raise ValueError("Scalerが学習されていません。")
            X_numerical_scaled = self.scaler.transform(X_numerical)

        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_feature_columns, index=X_numerical.index)

        X_operation_type_df = self._prepare_operation_type_features(data, is_training)

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
                raise ValueError("TF-IDF vectorizer (LongName)が学習されていません。")
            X_longname_tfidf = self.tfidf_vectorizer_longname.transform(longname_data)

        X_longname_tfidf_df = pd.DataFrame(X_longname_tfidf.toarray(),
                                          columns=[f'LongName_tfidf_{i}' for i in range(X_longname_tfidf.shape[1])],
                                          index=longname_data.index)

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
                raise ValueError("TF-IDF vectorizer (Parent)が学習されていません。")
            X_parent_tfidf = self.tfidf_vectorizer_parent.transform(parent_data)

        X_parent_tfidf_df = pd.DataFrame(X_parent_tfidf.toarray(),
                                        columns=[f'Parent_tfidf_{i}' for i in range(X_parent_tfidf.shape[1])],
                                        index=parent_data.index)

        X_parts = [X_numerical_scaled_df, X_longname_tfidf_df, X_parent_tfidf_df]
        if len(X_operation_type_df.columns) > 0:
            X_parts.append(X_operation_type_df)

        X = pd.concat(X_parts, axis=1)

        if is_training:
            self.all_feature_names = X.columns.tolist()
            print(f"初期使用特徴量数: {len(X.columns)}")
            print(f"  - 数値特徴量: {len(numerical_feature_columns)}")
            print(f"  - LongName TF-IDF: {X_longname_tfidf.shape[1]}")
            print(f"  - Parent TF-IDF: {X_parent_tfidf.shape[1]}")
            if len(X_operation_type_df.columns) > 0:
                print(f"  - operation_type One-Hot: {len(X_operation_type_df.columns)}")
            print(f"ラベル分布: 0={sum(y==0)}, 1={sum(y==1)}")

            self.original_class_distribution = {
                'class_0': sum(y==0),
                'class_1': sum(y==1),
                'total': len(y)
            }

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
            print(f"RandomUnderSamplerでエラー: {e}")
            print("元のデータを使用して続行します。")
            X_balanced = X.copy()
            y_balanced = y.copy()

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

            print(f"選択された特徴量数: {len(self.selected_features)}")

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
            print(f"特徴量選択でエラー: {e}")
            print("全ての特徴量を使用して続行します。")
            self.selected_features = X.columns.tolist()
            self.feature_importance_scores = np.ones(len(X.columns)) / len(X.columns)
            X_selected_df = X.copy()

        return X_selected_df

    def evaluate_model_with_cv(self, X: pd.DataFrame, y: pd.Series, k_folds: int = 10) -> dict:
        print(f"\n=== {k_folds}分割交差検証による評価 ===")

        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
        fold_results = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []
        }

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"フォールド {fold_idx + 1}/{k_folds} 処理中...", end=" ")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            X_train_fold_resampled, y_train_fold_resampled = self.apply_undersampling(X_train_fold, y_train_fold)

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
        for metric in fold_results.keys():
            scores = fold_results[metric]
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores

        return cv_results

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        print("\n=== 最終RandomForestモデル学習 ===")

        X_resampled, y_resampled = self.apply_undersampling(X, y)
        self.best_model = RandomForestClassifier(**self.default_rf_params)
        self.best_model.fit(X_resampled, y_resampled)

        print("最終RandomForestモデル学習完了")
        print(f"学習データ: {len(X_resampled)}件")

        return self.best_model

    def evaluate_on_test_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        print("\n=== テストデータでの最終モデル評価 ===")

        if self.best_model is None:
            raise ValueError("最終モデルが学習されていません。")

        y_test_pred = self.best_model.predict(X_test)
        y_test_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        test_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }

        print(f"テストデータサイズ: {len(X_test)}件")
        print(f"F1スコア: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"ROC-AUC: {test_results['roc_auc']:.4f}")

        self.predictions_data = {
            'y_true': y_test.values,
            'y_pred': y_test_pred,
            'y_pred_proba': y_test_pred_proba,
            'test_indices': self.test_indices
        }

        return test_results

    def save_model(self, model_path: str):
        model_data = {
            'model': self.best_model,
            'tfidf_vectorizer_longname': self.tfidf_vectorizer_longname,
            'tfidf_vectorizer_parent': self.tfidf_vectorizer_parent,
            'scaler': self.scaler,
            'operation_type_columns': self.operation_type_columns,
            'all_feature_names': self.all_feature_names,
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'has_operation_type': self.has_operation_type,
            'java_tokenizer': self.java_tokenizer,
            'default_rf_params': self.default_rf_params,
            'original_class_distribution': self.original_class_distribution,
            'resampled_train_distribution': self.resampled_train_distribution,
            'cv_results': self.cv_results,
            'test_results': self.test_results,
            'predictions_data': self.predictions_data
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"モデルを '{model_path}' に保存しました")

    def run_training_pipeline(self, data_path: str, max_rows: int = 10000, model_save_path: str = "bug_hunter_model.pkl"):
        print("=== BugHunter モデル訓練パイプライン ===")

        data = self.read_data(data_path, max_rows)

        X_full, y_full = self.prepare_data(data, is_training=True)

        print(f"\n=== 訓練・テストデータ分割 (テスト比率: {self.test_size}) ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=self.test_size,
            random_state=GLOBAL_SEED,
            stratify=y_full
        )

        self.test_indices = X_test.index.tolist()
        print(f"訓練データ: {len(X_train)}件")
        print(f"テストデータ: {len(X_test)}件")

        print("\n=== 特徴量選択器の学習 ===")
        X_train_temp_resampled, y_train_temp_resampled = self.apply_undersampling(X_train, y_train)
        self.select_features_by_rf_importance(X_train_temp_resampled, y_train_temp_resampled)

        X_train_reduced = X_train[self.selected_features]
        X_test_reduced = X_test[self.selected_features]

        self.cv_results = self.evaluate_model_with_cv(X_train_reduced, y_train, k_folds=10)

        self.train_final_model(X_train_reduced, y_train)

        self.test_results = self.evaluate_on_test_data(X_test_reduced, y_test)

        self.save_model(model_save_path)

        print("\n=== 10分割交差検証結果 ===")
        print(f"F1スコア: {self.cv_results['f1_mean']:.4f} ± {self.cv_results['f1_std']:.4f}")
        print(f"ROC-AUC: {self.cv_results['roc_auc_mean']:.4f} ± {self.cv_results['roc_auc_std']:.4f}")

        print("\n=== テストデータ評価結果 ===")
        print(f"F1スコア: {self.test_results['f1']:.4f}")
        print(f"ROC-AUC: {self.test_results['roc_auc']:.4f}")

        print("\n=== 訓練完了 ===")
        print(f"モデルが '{model_save_path}' に保存されました")

        return self.cv_results, self.test_results


def main():
    trainer = BugHunterTrainer(
        feature_selection_threshold=0.001,
        tfidf_max_features=100,
        java_tokenizer_min_length=2,
        include_package_tokens=False,
        test_size=0.2
    )

    try:
        data_path = "method-p_filtered_v2_changes.csv"
        model_save_path = "predictions_changes.pkl"

        cv_results, test_results = trainer.run_training_pipeline(
            data_path=data_path,
            max_rows=3000,
            model_save_path=model_save_path
        )

        print("\n" + "="*60)
        print("モデル訓練完了!")
        print("="*60)

    except FileNotFoundError:
        print(f"エラー: ファイル '{data_path}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
