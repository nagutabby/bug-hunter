import pandas as pd
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import pickle
import os
from typing import Dict, List, Optional, Tuple
import warnings
from train import JavaCodeTokenizer

warnings.filterwarnings('ignore')


class ComprehensiveBugHunterAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイル '{self.model_path}' が見つかりません")

        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        print(f"モデルを '{self.model_path}' から読み込みました")
        self._display_model_info()

    def _display_model_info(self):
        print("\n=== モデル基本情報 ===")
        print(f"モデルタイプ: {type(self.model_data['model']).__name__}")
        print(f"選択された特徴量数: {len(self.model_data['selected_features'])}")
        print(f"全特徴量数: {len(self.model_data['all_feature_names'])}")

        if self.model_data['cv_results']:
            cv_f1 = self.model_data['cv_results']['f1_mean']
            cv_f1_std = self.model_data['cv_results']['f1_std']
            print(f"交差検証F1スコア: {cv_f1:.4f} ± {cv_f1_std:.4f}")

        if self.model_data['test_results']:
            test_f1 = self.model_data['test_results']['f1']
            print(f"テストF1スコア: {test_f1:.4f}")

    def display_feature_importance_table(self, top_n: int = 10) -> pd.DataFrame:
        if not self.model_data['feature_importance_scores'] is not None:
            print("Feature Importanceスコアが見つかりません。")
            return None

        all_features_df = pd.DataFrame({
            '特徴量': self.model_data['all_feature_names'],
            'Feature Importance': self.model_data['feature_importance_scores']
        }).sort_values('Feature Importance', ascending=False)

        selected_features_df = all_features_df[
            all_features_df['特徴量'].isin(self.model_data['selected_features'])
        ]

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
        original_dist = self.model_data.get('original_class_distribution')
        resampled_dist = self.model_data.get('resampled_train_distribution')

        if not original_dist or not resampled_dist:
            print("サンプリング情報が見つかりません。")
            return

        print("\n=== RandomUnderSampler サマリー ===")
        print(f"元データ (訓練データ):")
        print(f"  クラス 0: {original_dist['class_0']:,}件")
        print(f"  クラス 1: {original_dist['class_1']:,}件")
        print(f"  合計: {original_dist['total']:,}件")

        print(f"\nアンダーサンプリング適用後 (訓練データ):")
        print(f"  クラス 0: {resampled_dist['class_0']:,}件")
        print(f"  クラス 1: {resampled_dist['class_1']:,}件")
        print(f"  合計: {resampled_dist['total']:,}件")

        change_rate = (resampled_dist['total'] / original_dist['total'] - 1) * 100
        print(f"\n元の訓練データからの変化率: {change_rate:.1f}%")

    def display_feature_selection_summary(self):
        feature_scores = self.model_data.get('feature_importance_scores')
        selected_features = self.model_data.get('selected_features')
        all_features = self.model_data.get('all_feature_names')

        if not feature_scores is not None or not selected_features or not all_features:
            print("特徴量選択情報が見つかりません。")
            return

        print(f"\n=== RandomForest Feature Importanceによる特徴量選択サマリー ===")
        print(f"特徴量選択手法: RandomForest Feature Importance")
        print(f"元の特徴量数: {len(all_features)}")
        print(f"選択された特徴量数: {len(selected_features)}")
        print(f"選択率: {len(selected_features)/len(all_features)*100:.1f}%")

        print(f"Feature Importance統計:")
        print(f"  最大値: {np.max(feature_scores):.4f}")
        print(f"  平均値: {np.mean(feature_scores):.4f}")
        print(f"  最小値: {np.min(feature_scores):.4f}")

    def display_operation_type_analysis(self):
        has_operation_type = self.model_data.get('has_operation_type', False)
        operation_type_columns = self.model_data.get('operation_type_columns')

        if not has_operation_type:
            print("\noperation_typeカラムは存在しません。")
            return

        print(f"\n=== operation_type One-Hotエンコーディング分析 ===")
        if operation_type_columns:
            print(f"生成されたoperation_typeカラム:")
            for i, col in enumerate(operation_type_columns):
                print(f"  {i+1}. {col}")

            selected_features = self.model_data.get('selected_features', [])
            selected_op_features = [f for f in selected_features if f.startswith('operation_type_')]
            print(f"\n選択されたoperation_type特徴量: {len(selected_op_features)}個")

            feature_scores = self.model_data.get('feature_importance_scores')
            all_features = self.model_data.get('all_feature_names')

            if feature_scores is not None and all_features:
                for feature in selected_op_features:
                    try:
                        idx = all_features.index(feature)
                        importance = feature_scores[idx]
                        print(f"  {feature}: {importance:.4f}")
                    except (ValueError, IndexError):
                        print(f"  {feature}: 重要度不明")
        else:
            print("operation_typeカラム情報が取得できませんでした。")

    def get_cv_detailed_results(self) -> Optional[pd.DataFrame]:
        cv_results = self.model_data.get('cv_results')
        if not cv_results:
            print("交差検証結果が見つかりません。")
            return None

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

    def _get_real_data_statistics(self, data_path: str, max_rows: int = 10000) -> Optional[pd.DataFrame]:
        if not os.path.exists(data_path):
            print(f"データファイル '{data_path}' が見つかりません")
            return None

        print(f"実データから統計情報を取得中... (最大{max_rows}行)")
        try:
            data = pd.read_csv(data_path, nrows=max_rows)
            print(f"データ読み込み完了: {len(data)}行")

            from train import BugHunterTrainer
            temp_trainer = BugHunterTrainer()

            temp_trainer.tfidf_vectorizer_longname = self.model_data['tfidf_vectorizer_longname']
            temp_trainer.tfidf_vectorizer_parent = self.model_data['tfidf_vectorizer_parent']
            temp_trainer.operation_type_columns = self.model_data['operation_type_columns']
            temp_trainer.all_feature_names = self.model_data['all_feature_names']
            temp_trainer.has_operation_type = self.model_data['has_operation_type']
            temp_trainer.java_tokenizer = self.model_data['java_tokenizer']

            X_processed, _ = temp_trainer.prepare_data(data, is_training=False)
            print(f"前処理完了: {X_processed.shape[1]}個の特徴量")

            return X_processed

        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None

    def _calculate_feature_range(self, feature_data: pd.Series, feature_name: str) -> Tuple[float, float, str]:
        feature_data_clean = feature_data.dropna()

        if len(feature_data_clean) == 0:
            return 0, 1, "no_data"

        unique_vals = feature_data_clean.unique()
        if len(unique_vals) <= 2 and all(val in [0, 1] for val in unique_vals):
            return 0, 1, "binary"

        # 外れ値を除外した範囲を計算（5%～95%パーセンタイル）
        if feature_name.startswith('LongName_tfidf_') or feature_name.startswith('Parent_tfidf_'):
            # TF-IDFは0以上なので、下限は0に固定
            min_val = 0
            max_val = feature_data_clean.quantile(0.95)
            range_type = "tfidf"
        else:
            # その他の特徴量は5%～95%のパーセンタイルを使用
            min_val = feature_data_clean.quantile(0.05)
            max_val = feature_data_clean.quantile(0.95)
            range_type = "continuous"

        # 範囲が非常に小さい場合の処理
        if abs(max_val - min_val) < 1e-10:
            mean_val = feature_data_clean.mean()
            std_val = feature_data_clean.std() if feature_data_clean.std() > 0 else 1
            min_val = mean_val - 2 * std_val
            max_val = mean_val + 2 * std_val
            range_type = "expanded"

        return min_val, max_val, range_type

    def _generate_realistic_sample_data(self, X_real: pd.DataFrame, selected_features: List[str],
                                      n_samples: int = 1000) -> pd.DataFrame:
        print("実データの統計に基づいたサンプルデータを生成中...")

        sample_data = {}

        for feature in selected_features:
            if feature not in X_real.columns:
                if feature.startswith('operation_type_'):
                    sample_data[feature] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
                else:
                    sample_data[feature] = np.random.normal(0, 1, n_samples)
                continue

            feature_data = X_real[feature].dropna()

            if len(feature_data) == 0:
                sample_data[feature] = np.zeros(n_samples)
                continue

            if feature.startswith('operation_type_'):
                prob_1 = feature_data.mean()
                sample_data[feature] = np.random.choice([0, 1], n_samples,
                                                       p=[1-prob_1, prob_1])

            elif feature.startswith('LongName_tfidf_') or feature.startswith('Parent_tfidf_'):
                mean_val = feature_data.mean()
                if mean_val > 0:
                    var_val = feature_data.var()
                    if var_val > 0:
                        scale = var_val / mean_val
                        shape = mean_val / scale
                        sample_data[feature] = np.random.gamma(shape, scale, n_samples)
                        sample_data[feature] = np.maximum(0, sample_data[feature])
                    else:
                        sample_data[feature] = np.full(n_samples, mean_val)
                else:
                    sample_data[feature] = np.zeros(n_samples)

            else:
                mean_val = feature_data.mean()
                std_val = feature_data.std()
                if std_val > 0:
                    sample_data[feature] = np.random.normal(mean_val, std_val, n_samples)
                else:
                    sample_data[feature] = np.full(n_samples, mean_val)

        return pd.DataFrame(sample_data)

    def plot_feature_histograms(self, data_path: str, top_n: int = 10,
                               save_path: Optional[str] = None, max_rows: int = 10000):
        print(f"\n=== 上位{top_n}特徴量のヒストグラム分析 ===")

        X_processed = self._get_real_data_statistics(data_path, max_rows)
        if X_processed is None:
            return

        feature_scores = self.model_data['feature_importance_scores']
        all_features = self.model_data['all_feature_names']
        selected_features = self.model_data['selected_features']

        selected_features_df = pd.DataFrame({
            '特徴量': all_features,
            'Feature Importance': feature_scores
        })

        selected_features_df = selected_features_df[
            selected_features_df['特徴量'].isin(selected_features)
        ].sort_values('Feature Importance', ascending=False)

        top_features = selected_features_df.head(top_n)['特徴量'].tolist()

        print(f"ヒストグラム（上位{len(top_features)}特徴量）")
        for i, feature in enumerate(top_features, 1):
            importance_idx = all_features.index(feature)
            importance = feature_scores[importance_idx]
            feature_type = self._get_feature_type(feature)
            print(f"  {i:2d}. {feature} ({feature_type}) - 重要度: {importance:.4f}")

        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font='IPAexGothic')

        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

        n_cols = 4
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig_width = n_cols * 5
        fig_height = n_rows * 4

        print(f"ヒストグラム描画中... ({n_rows}行 × {n_cols}列のグリッド)")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, feature_name in enumerate(top_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            try:
                if feature_name in X_processed.columns:
                    feature_data = X_processed[feature_name].dropna()

                    if len(feature_data) == 0:
                        self._plot_no_data_message(ax, feature_name)
                        continue

                    if feature_name.startswith('operation_type_'):
                        self._plot_binary_histogram(ax, feature_data, feature_name)
                    else:
                        self._plot_continuous_histogram(ax, feature_data, feature_name)

                    short_name = self._shorten_feature_name(feature_name, 20)
                    importance_idx = all_features.index(feature_name)
                    importance = feature_scores[importance_idx]
                    ax.set_title(f'{short_name}', fontsize=11, pad=10)

                else:
                    self._plot_no_data_message(ax, feature_name)

            except Exception as e:
                print(f"特徴量 '{feature_name}' のヒストグラム描画でエラー: {e}")
                self._plot_error_message(ax, feature_name, str(e))

        for i in range(len(top_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is None:
            save_path = "feature_histograms.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"特徴量ヒストグラムを '{save_path}' に保存しました")

        plt.show()
        print("特徴量ヒストグラム描画完了")

    def _plot_binary_histogram(self, ax, feature_data: pd.Series, feature_name: str):
        unique_values = sorted(feature_data.unique())
        value_counts = feature_data.value_counts().sort_index()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(range(len(unique_values)),
                     [value_counts.get(val, 0) for val in unique_values],
                     color=colors[:len(unique_values)], alpha=0.7, edgecolor='navy')

        ax.set_xticks(range(len(unique_values)))
        ax.set_xticklabels([f'{val}' for val in unique_values])
        ax.set_xlabel('値')
        ax.set_ylabel('頻度')

        for bar, count in zip(bars, [value_counts.get(val, 0) for val in unique_values]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(value_counts) * 0.01,
                   f'{count}', ha='center', va='bottom', fontsize=9)

        total_count = len(feature_data)
        proportions = [value_counts.get(val, 0) / total_count for val in unique_values]

        stats_text = f'N={total_count}\n'
        for val, prop in zip(unique_values, proportions):
            stats_text += f'{val}: {prop:.1%}\n'

        ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.grid(True, alpha=0.3, axis='y')

    def _plot_continuous_histogram(self, ax, feature_data: pd.Series, feature_name: str):
        mean_val = feature_data.mean()
        median_val = feature_data.median()

        n_bins = min(50, max(10, len(feature_data) // 20))

        counts, bins, patches = ax.hist(feature_data, bins=n_bins, density=True,
                                       alpha=0.7, color='skyblue', edgecolor='navy')

        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label='平均')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label='中央値')

        ax.set_xlabel('値')
        ax.set_ylabel('密度')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def plot_feature_importance_chart(self, top_n: int = 10, save_path: Optional[str] = None):
        feature_scores = self.model_data['feature_importance_scores']
        selected_features = self.model_data['selected_features']
        all_features = self.model_data['all_feature_names']

        selected_features_df = pd.DataFrame({
            '特徴量': all_features,
            'Feature Importance': feature_scores
        })

        selected_features_df = selected_features_df[
            selected_features_df['特徴量'].isin(selected_features)
        ].sort_values('Feature Importance', ascending=False)

        top_features = selected_features_df.head(top_n)['特徴量'].tolist()

        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font='IPAexGothic')

        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

        importances = []
        feature_types = []
        short_names = []

        for feature in top_features:
            idx = all_features.index(feature)
            importances.append(feature_scores[idx])
            feature_types.append(self._get_feature_type(feature))
            short_names.append(self._shorten_feature_name(feature))

        type_colors = {
            'LongName TF-IDF': '#FF6B6B',
            'Parent TF-IDF': '#4ECDC4',
            'operation_type': '#45B7D1',
            '数値': '#96CEB4'
        }

        colors = [type_colors.get(ft, '#95A5A6') for ft in feature_types]

        fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.4)))

        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names)
        ax.set_xlabel('Feature Importance', fontsize=12)

        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center', fontsize=9)

        unique_types = list(set(feature_types))
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=type_colors.get(t, '#95A5A6'),
                                       alpha=0.8, edgecolor='black') for t in unique_types]
        ax.legend(legend_elements, unique_types, loc='lower right', fontsize=10)

        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        plt.tight_layout()

        if save_path is None:
            save_path = "feature_importance_chart.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"特徴量重要度チャートを '{save_path}' に保存しました")

        plt.show()
        print("特徴量重要度チャート描画完了")

    def plot_partial_dependence(self, data_path: str, top_n: int = 10,
                               save_path: Optional[str] = None, max_rows: int = 10000):
        print(f"\n=== Partial Dependence Plots (上位{top_n}特徴量) ===")

        X_real = self._get_real_data_statistics(data_path, max_rows)
        if X_real is None:
            print("実データの取得に失敗しました")
            return

        model = self.model_data['model']
        feature_scores = self.model_data['feature_importance_scores']
        selected_features = self.model_data['selected_features']
        all_features = self.model_data['all_feature_names']

        if not model or feature_scores is None or not selected_features:
            print("PDP描画に必要なデータが不足しています。")
            return

        selected_features_df = pd.DataFrame({
            '特徴量': all_features,
            'Feature Importance': feature_scores
        })

        selected_features_df = selected_features_df[
            selected_features_df['特徴量'].isin(selected_features)
        ].sort_values('Feature Importance', ascending=False)

        top_features = selected_features_df.head(top_n)['特徴量'].tolist()

        print(f"Partial Dependence Plots（上位{len(top_features)}特徴量）")
        for i, feature in enumerate(top_features, 1):
            importance_idx = all_features.index(feature)
            importance = feature_scores[importance_idx]
            feature_type = self._get_feature_type(feature)
            print(f"  {i:2d}. {feature} ({feature_type}) - 重要度: {importance:.4f}")

        X_sample = self._generate_realistic_sample_data(X_real, selected_features, n_samples=1000)

        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font='IPAexGothic')

        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

        n_cols = 4
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig_width = n_cols * 4
        fig_height = n_rows * 3

        print(f"PDP描画中... ({n_rows}行 × {n_cols}列のグリッド)")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, feature_name in enumerate(top_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            try:
                feature_idx = selected_features.index(feature_name)

                if feature_name in X_real.columns:
                    min_val, max_val, range_type = self._calculate_feature_range(
                        X_real[feature_name], feature_name
                    )

                    print(f"  {feature_name}: 範囲 [{min_val:.3f}, {max_val:.3f}] ({range_type})")

                    if range_type == "binary":
                        self._plot_binary_pdp(ax, feature_name, feature_idx, X_sample, model,
                                            min_val, max_val)
                    else:
                        self._plot_continuous_pdp(ax, feature_name, feature_idx, X_sample, model,
                                                min_val, max_val, range_type)
                else:
                    print(f"  {feature_name}: 実データに存在しません")
                    self._plot_no_data_message(ax, feature_name)

                short_name = self._shorten_feature_name(feature_name)
                importance_idx = all_features.index(feature_name)
                importance = feature_scores[importance_idx]
                ax.set_title(f'{short_name}', fontsize=10, pad=10)

                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.set_ylabel('Partial Dependence', fontsize=9)

            except Exception as e:
                print(f"特徴量 '{feature_name}' のPDP描画でエラー: {e}")
                self._plot_error_message(ax, feature_name, str(e))

        for i in range(len(top_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is None:
            save_path = "partial_dependence_plots.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Partial Dependence Plotsを '{save_path}' に保存しました")

        plt.show()
        print("Partial Dependence Plots描画完了")

    def _plot_binary_pdp(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame,
                        model, min_val: float, max_val: float):
        try:
            unique_values = [0, 1]
            pdp_values = []

            for value in unique_values:
                X_temp = X_sample.copy()
                X_temp.iloc[:, feature_idx] = value
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_value = np.mean(predictions)
                pdp_values.append(pdp_value)

            colors = ['#FF6B6B', '#4ECDC4']
            bars = ax.bar(range(len(unique_values)), pdp_values,
                         color=colors, alpha=0.7, edgecolor='navy', linewidth=1)

            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([f'{val}' for val in unique_values])
            ax.set_xlabel('値')

            for i, (bar, pdp_val) in enumerate(zip(bars, pdp_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(pdp_values) * 0.01,
                       f'{pdp_val:.3f}', ha='center', va='bottom', fontsize=8)

            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        except Exception as e:
            print(f"  バイナリPDP描画でエラー: {e}")
            self._plot_error_message(ax, feature_name, str(e))

    def _plot_continuous_pdp(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame,
                           model, min_val: float, max_val: float, range_type: str):
        try:
            if range_type == "tfidf":
                grid_values = np.linspace(0, max_val, 30)
            else:
                grid_values = np.linspace(min_val, max_val, 30)

            pdp_values = []
            for grid_val in grid_values:
                X_temp = X_sample.copy()
                X_temp.iloc[:, feature_idx] = grid_val
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_values.append(np.mean(predictions))

            ax.plot(grid_values, pdp_values, 'o-', color='blue', linewidth=2, markersize=3)
            ax.set_xlabel('特徴量の値')
            ax.set_xlim(min_val, max_val)

            range_info = f'範囲: [{min_val:.2f}, {max_val:.2f}]'
            ax.text(0.05, 0.95, range_info, transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"  連続値PDP描画でエラー: {e}")
            self._plot_error_message(ax, feature_name, str(e))

    def _plot_no_data_message(self, ax, feature_name: str):
        ax.text(0.5, 0.5, f'{self._shorten_feature_name(feature_name)}\n\nデータなし',
               ha='center', va='center', transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE4B5", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_error_message(self, ax, feature_name: str, error_msg: str):
        ax.text(0.5, 0.5, f'エラー:\n{self._shorten_feature_name(feature_name)}\n\n{error_msg[:30]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def _get_feature_type(self, feature_name: str) -> str:
        if feature_name.startswith('LongName_tfidf_'):
            return 'LongName TF-IDF'
        elif feature_name.startswith('Parent_tfidf_'):
            return 'Parent TF-IDF'
        elif feature_name.startswith('operation_type_'):
            return 'operation_type'
        else:
            return '数値'

    def _shorten_feature_name(self, feature_name: str, max_length: int = 25) -> str:
        if len(feature_name) <= max_length:
            return feature_name

        if feature_name.startswith('LongName_tfidf_'):
            return f"LN_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('Parent_tfidf_'):
            return f"P_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('operation_type_'):
            op_type = feature_name.replace('operation_type_', '')
            return f"op_{op_type[:15]}"
        else:
            if len(feature_name) > max_length:
                return feature_name[:max_length-3] + "..."
            return feature_name

    def plot_cv_performance_chart(self, save_path: Optional[str] = None):
        cv_results = self.model_data.get('cv_results')
        if not cv_results:
            print("交差検証結果が見つかりません。")
            return

        print("\n=== 交差検証パフォーマンスチャート描画 ===")

        metrics = ['f1_scores', 'precision_scores', 'recall_scores', 'accuracy_scores', 'roc_auc_scores']
        metric_names = ['F1', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']

        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font='IPAexGothic')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            scores = cv_results[metric]
            folds = list(range(1, len(scores) + 1))

            ax.plot(folds, scores, 'o-', linewidth=2, markersize=6, label=name)
            ax.axhline(y=np.mean(scores), color='red', linestyle='--', alpha=0.7,
                      label=f'平均: {np.mean(scores):.3f}')

            ax.fill_between(folds,
                           np.mean(scores) - np.std(scores),
                           np.mean(scores) + np.std(scores),
                           alpha=0.2, color='gray', label=f'±1σ: {np.std(scores):.3f}')

            ax.set_xlabel('Fold')
            ax.set_ylabel(f'{name} Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(folds)

        axes[5].set_visible(False)

        plt.tight_layout()

        if save_path is None:
            save_path = "cv_performance_chart.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"交差検証パフォーマンスチャートを '{save_path}' に保存しました")

        plt.show()
        print("交差検証パフォーマンスチャート描画完了")

    def generate_comprehensive_analysis_report(self, data_path: Optional[str] = None, output_dir: str = "."):
        print("="*80)
        print("BugHunter 統合分析レポート")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        self._display_model_info()
        self.display_sampling_summary()
        self.display_feature_selection_summary()
        self.display_operation_type_analysis()
        feature_df = self.display_feature_importance_table(top_n=10)

        cv_df = self.get_cv_detailed_results()
        if cv_df is not None:
            print("\n=== 各フォールドの詳細結果（交差検証） ===")
            print(cv_df.round(4))

        print(f"\n{'='*80}")
        print("グラフ生成中...")
        print(f"{'='*80}")

        importance_path = os.path.join(output_dir, "feature_importance_chart.png")
        self.plot_feature_importance_chart(top_n=10, save_path=importance_path)

        cv_performance_path = os.path.join(output_dir, "cv_performance_chart.png")
        self.plot_cv_performance_chart(save_path=cv_performance_path)

        if data_path and os.path.exists(data_path):
            print(f"\n{'='*80}")
            print("実データを使用した分析")
            print(f"{'='*80}")

            histogram_path = os.path.join(output_dir, "feature_histograms.png")
            self.plot_feature_histograms(
                data_path=data_path,
                top_n=10,
                save_path=histogram_path,
                max_rows=5000
            )

            pdp_path = os.path.join(output_dir, "partial_dependence_plots.png")
            self.plot_partial_dependence(
                data_path=data_path,
                top_n=10,
                save_path=pdp_path,
                max_rows=5000
            )

        else:
            if data_path:
                print(f"\nデータファイル '{data_path}' が見つからないため、実データ分析をスキップします")
            else:
                print(f"\ndata_pathが指定されていないため、実データ分析をスキップします")

        print("\n" + "="*80)
        print("統合分析レポート完了")
        print(f"出力ディレクトリ: {output_dir}")
        print("生成されたファイル:")
        print("  - feature_importance_chart.png (特徴量重要度)")
        print("  - cv_performance_chart.png (交差検証パフォーマンス)")
        if data_path and os.path.exists(data_path):
            print("  - feature_histograms.png (特徴量ヒストグラム)")
            print("  - partial_dependence_plots.png (PDP)")
        print("="*80)

def main():
    try:
        project_name = "orientdb"
        base_dir = f"../data/remove/{project_name}/"
        output_dir = f"../materials/images/{project_name}/"

        analyzer = ComprehensiveBugHunterAnalyzer(base_dir + "predictions_add_method_commit_level_metrics.pkl")
        data_path = base_dir + "method-p_add_method_commit_level_metrics.csv"

        os.makedirs(output_dir, exist_ok=True)

        analyzer.generate_comprehensive_analysis_report(data_path=data_path, output_dir=output_dir)

        print("\n" + "="*60)
        print("全ての分析とグラフ生成が完了しました！")
        print("="*60)

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("まず bug_hunter_trainer.py を実行してモデルを訓練してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
