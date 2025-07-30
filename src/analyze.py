import pandas as pd
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import pickle
import os
from typing import Dict, List, Optional
import warnings
from train import JavaCodeTokenizer

warnings.filterwarnings('ignore')


class BugHunterAnalyzer:
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

    def display_feature_importance_table(self, top_n: int = 20) -> pd.DataFrame:
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

    def plot_feature_histograms(self, data_path: str, top_n: int = 20,
                               save_path: Optional[str] = None, max_rows: int = 10000):
        print(f"\n=== 上位{top_n}特徴量のヒストグラム分析 ===")

        if not os.path.exists(data_path):
            print(f"データファイル '{data_path}' が見つかりません")
            return

        print(f"データを読み込み中... (最大{max_rows}行)")
        try:
            data = pd.read_csv(data_path, nrows=max_rows)
            print(f"データ読み込み完了: {len(data)}行")
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return

        if not self.model_data:
            print("モデルデータが読み込まれていません")
            return

        try:
            from train import BugHunterTrainer
            temp_trainer = BugHunterTrainer()

            temp_trainer.tfidf_vectorizer_longname = self.model_data['tfidf_vectorizer_longname']
            temp_trainer.tfidf_vectorizer_parent = self.model_data['tfidf_vectorizer_parent']
            temp_trainer.scaler = self.model_data['scaler']
            temp_trainer.operation_type_columns = self.model_data['operation_type_columns']
            temp_trainer.all_feature_names = self.model_data['all_feature_names']
            temp_trainer.has_operation_type = self.model_data['has_operation_type']
            temp_trainer.java_tokenizer = self.model_data['java_tokenizer']

            X_processed, _ = temp_trainer.prepare_data(data, is_training=False)
            print(f"前処理完了: {X_processed.shape[1]}個の特徴量")

        except Exception as e:
            print(f"データ前処理エラー: {e}")
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

        print(f"ヒストグラム対象特徴量（上位{len(top_features)}個）:")
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
                    ax.set_title(f'{short_name}\n重要度: {importance:.3f}', fontsize=11, pad=10)

                else:
                    self._plot_no_data_message(ax, feature_name)

            except Exception as e:
                print(f"特徴量 '{feature_name}' のヒストグラム描画でエラー: {e}")
                self._plot_error_message(ax, feature_name, str(e))

        for i in range(len(top_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(f'特徴量分布ヒストグラム (上位{len(top_features)}特徴量)', fontsize=16, y=0.98)

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

        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8)
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8)

        ax.set_xlabel('値')
        ax.set_ylabel('密度')
        ax.grid(True, alpha=0.3)

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

    def plot_partial_dependence(self, top_n: int = 20, save_path: Optional[str] = None):
        model = self.model_data['model']
        feature_scores = self.model_data['feature_importance_scores']
        selected_features = self.model_data['selected_features']
        all_features = self.model_data['all_feature_names']

        if not model or feature_scores is None or not selected_features:
            print("PDP描画に必要なデータが不足しています。")
            return

        print(f"\n=== 特徴量上位{top_n}個の分析と可視化 ===")

        selected_features_df = pd.DataFrame({
            '特徴量': all_features,
            'Feature Importance': feature_scores
        })

        selected_features_df = selected_features_df[
            selected_features_df['特徴量'].isin(selected_features)
        ].sort_values('Feature Importance', ascending=False)

        top_features = selected_features_df.head(top_n)['特徴量'].tolist()

        print(f"対象特徴量（上位{len(top_features)}個）:")
        for i, feature in enumerate(top_features, 1):
            importance_idx = all_features.index(feature)
            importance = feature_scores[importance_idx]
            feature_type = self._get_feature_type(feature)
            print(f"  {i:2d}. {feature} ({feature_type}) - 重要度: {importance:.4f}")

        print("\n=== 1) 特徴量重要度チャート描画 ===")
        importance_save_path = "feature_importance_chart.png" if save_path is None else save_path.replace('.png', '_importance.png')
        self._plot_feature_importance_chart(top_features, feature_scores, all_features, importance_save_path)

        print("\n=== 2) Partial Dependence Plots描画 ===")
        pdp_save_path = "partial_dependence_plots.png" if save_path is None else save_path
        self._plot_partial_dependence_plots(top_features, model, selected_features, pdp_save_path)

    def _plot_partial_dependence_plots(self, top_features: List[str], model, selected_features: List[str], save_path: str):
        try:
            print("Partial Dependence Plot用のサンプルデータを生成中...")

            n_samples = 1000
            sample_data = {}

            for feature in selected_features:
                if feature.startswith('LongName_tfidf_') or feature.startswith('Parent_tfidf_'):
                    sample_data[feature] = np.random.uniform(-50, 50, n_samples)
                elif feature.startswith('operation_type_'):
                    sample_data[feature] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
                else:
                    sample_data[feature] = np.random.uniform(-50, 50, n_samples)

            X_sample = pd.DataFrame(sample_data)

            sns.set_style("whitegrid")
            sns.set_palette("husl")
            sns.set(font='IPAexGothic')

            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['figure.titlesize'] = 16

            n_cols = 5
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

                    if feature_name.startswith('operation_type_'):
                        self._plot_operation_type_pdp(ax, feature_name, feature_idx, X_sample, model)
                    else:
                        try:
                            grid_values = np.linspace(-50, 50, 30)

                            pdp_values = []
                            for grid_val in grid_values:
                                X_temp = X_sample.copy()
                                X_temp.iloc[:, feature_idx] = grid_val
                                predictions = model.predict_proba(X_temp)[:, 1]
                                pdp_values.append(np.mean(predictions))

                            ax.plot(grid_values, pdp_values, 'o-', color='blue', linewidth=2, markersize=3)
                            ax.set_xlabel('特徴量の値', fontsize=9)
                            ax.set_xlim(-50, 50)

                        except Exception as pdp_error:
                            print(f"PDP手動計算でエラー: {pdp_error}")
                            self._plot_manual_pdp_simple(ax, feature_name, feature_idx, X_sample, model)

                    short_name = self._shorten_feature_name(feature_name)
                    all_features = self.model_data['all_feature_names']
                    feature_scores = self.model_data['feature_importance_scores']
                    importance_idx = all_features.index(feature_name)
                    importance = feature_scores[importance_idx]
                    ax.set_title(f'{short_name}\n(重要度: {importance:.3f})', fontsize=10, pad=10)

                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.set_ylabel('Partial Dependence', fontsize=9)

                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

                except Exception as e:
                    print(f"特徴量 '{feature_name}' のPDP描画でエラー: {e}")
                    self._plot_error_alternative(ax, feature_name, e)

            for i in range(len(top_features), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)

            fig.suptitle(f'Partial Dependence Plots (上位{len(top_features)}特徴量) \n特徴量範囲: -50 ～ 50', fontsize=16, y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Partial Dependence Plotsを '{save_path}' に保存しました")

            plt.show()
            print("Partial Dependence Plots描画完了")

        except Exception as e:
            print(f"PDP描画中にエラーが発生しました: {e}")
            print("PDPの描画に失敗しました。")

    def _plot_operation_type_pdp(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame, model):
        try:
            feature_data = X_sample.iloc[:, feature_idx]
            unique_values = sorted(feature_data.unique())

            print(f"  {feature_name} のユニーク値: {unique_values}")

            if len(unique_values) < 2 or 0 not in unique_values or 1 not in unique_values:
                print(f"  {feature_name} に0と1の両方を強制追加")
                X_modified = X_sample.copy()

                n_half = len(X_modified) // 2
                X_modified.iloc[:n_half, feature_idx] = 0
                X_modified.iloc[n_half:, feature_idx] = 1

                feature_data = X_modified.iloc[:, feature_idx]
                unique_values = [0, 1]
            else:
                X_modified = X_sample

            pdp_values = []

            for value in unique_values:
                X_temp = X_modified.copy()
                X_temp.iloc[:, feature_idx] = value

                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_value = np.mean(predictions)
                pdp_values.append(pdp_value)

            colors = ['#FF6B6B' if val == 0 else '#4ECDC4' for val in unique_values]
            bars = ax.bar(range(len(unique_values)), pdp_values,
                         color=colors, alpha=0.7, edgecolor='navy', linewidth=1)

            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([f'{val}' for val in unique_values])

            for i, (bar, pdp_val) in enumerate(zip(bars, pdp_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(pdp_values) * 0.01,
                       f'{pdp_val:.3f}', ha='center', va='bottom', fontsize=8)

            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            print(f"  {feature_name} PDP値: {dict(zip(unique_values, pdp_values))}")

        except Exception as e:
            print(f"  operation_type PDP描画でエラー: {e}")
            self._plot_simple_bar_alternative(ax, feature_name)

    def _plot_manual_pdp_simple(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame, model):
        try:
            feature_data = X_sample.iloc[:, feature_idx]

            min_val, max_val = feature_data.min(), feature_data.max()

            if abs(max_val - min_val) < 1e-10:
                self._plot_simple_bar_alternative(ax, feature_name)
                return

            if len(feature_data.unique()) <= 10:
                test_values = sorted(feature_data.unique())
            else:
                test_values = np.linspace(min_val, max_val, 20)

            pdp_values = []
            for value in test_values:
                X_temp = X_sample.copy()
                X_temp.iloc[:, feature_idx] = value
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_values.append(np.mean(predictions))

            if len(test_values) <= 10:
                ax.bar(range(len(test_values)), pdp_values, alpha=0.7, color='skyblue', edgecolor='navy')
                ax.set_xticks(range(len(test_values)))
                ax.set_xticklabels([f'{val:.2f}' for val in test_values])
            else:
                ax.plot(test_values, pdp_values, 'o-', color='blue', linewidth=2, markersize=4)

            ax.grid(True, alpha=0.3, linestyle='--')

        except Exception as e:
            print(f"  手動PDP描画でエラー: {e}")
            self._plot_simple_bar_alternative(ax, feature_name)

    def _plot_simple_bar_alternative(self, ax, feature_name: str):
        ax.text(0.5, 0.5, f'{self._shorten_feature_name(feature_name)}\n\nPDP計算困難\n(単一値または\nデータ不足)',
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", alpha=0.8, edgecolor='blue'))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#f8f8f8')

        ax.set_ylabel('Partial Dependence', fontsize=9)

    def _plot_error_alternative(self, ax, feature_name: str, error):
        ax.text(0.5, 0.5, f'描画エラー:\n{self._shorten_feature_name(feature_name)}\n\n{str(error)[:50]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8, edgecolor='red'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Partial Dependence', fontsize=9)
        ax.set_facecolor('#f8f8f8')

    def _get_feature_type(self, feature_name: str) -> str:
        if feature_name.startswith('LongName_tfidf_'):
            return 'LongName TF-IDF'
        elif feature_name.startswith('Parent_tfidf_'):
            return 'Parent TF-IDF'
        elif feature_name.startswith('operation_type_'):
            return 'operation_type'
        else:
            return '数値'

    def _plot_feature_importance_chart(self, top_features: List[str], feature_scores: np.ndarray,
                                     all_features: List[str], save_path: Optional[str] = None):
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
        ax.set_title(f'特徴量重要度 Top {len(top_features)}', fontsize=14, pad=20)

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

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"特徴量重要度チャートを '{save_path}' に保存しました")

        plt.show()
        print("特徴量重要度チャート描画完了")

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

    def generate_analysis_report(self, data_path: Optional[str] = None, output_dir: str = "."):
        print("="*80)
        print("BugHunter モデル分析レポート")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        self._display_model_info()
        self.display_sampling_summary()
        self.display_feature_selection_summary()
        self.display_operation_type_analysis()
        feature_df = self.display_feature_importance_table(top_n=20)

        cv_df = self.get_cv_detailed_results()
        if cv_df is not None:
            print("\n=== 各フォールドの詳細結果（交差検証） ===")
            print(cv_df.round(4))

        if data_path and os.path.exists(data_path):
            print(f"\n{'='*80}")
            print("実データを使用した特徴量分布分析")
            print(f"{'='*80}")

            histogram_path = os.path.join(output_dir, "feature_histograms.png")
            self.plot_feature_histograms(
                data_path=data_path,
                top_n=20,
                save_path=histogram_path,
                max_rows=3000
            )

        else:
            if data_path:
                print(f"\nデータファイル '{data_path}' が見つからないため、ヒストグラム分析をスキップします")
            else:
                print(f"\ndata_pathが指定されていないため、ヒストグラム分析をスキップします")

        print("\n" + "="*80)
        print("分析レポート完了")
        print("="*80)


def main():
    try:
        base_dir = "../data/remove/elasticsearch/"
        output_dir = "../materials/images/elasticsearch"
        analyzer = BugHunterAnalyzer(base_dir + "predictions_add_change_metrics.pkl")
        data_path = base_dir + "method-p_add_change_metrics.csv"

        os.makedirs(output_dir, exist_ok=True)

        analyzer.generate_analysis_report(data_path=data_path, output_dir=output_dir)

        pdp_save_path = os.path.join(output_dir, "analysis_charts.png")
        analyzer.plot_partial_dependence(
            top_n=20,
            save_path=pdp_save_path
        )

        print("\n" + "="*60)
        print(f"分析完了！{output_dir} に以下のファイルが生成されました:")
        print("  - feature_histograms.png (特徴量ヒストグラム)")
        print("  - feature_importance_chart.png (特徴量重要度)")
        print("  - partial_dependence_plots.png (Partial Dependence Plots)")
        print("="*60)

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("まず bug_hunter_trainer.py を実行してモデルを訓練してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
