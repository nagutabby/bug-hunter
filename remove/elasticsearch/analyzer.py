import pandas as pd
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語フォント対応
import pickle
import os
from typing import Dict, List, Optional
import warnings
from trainer import JavaCodeTokenizer

warnings.filterwarnings('ignore')


class BugHunterAnalyzer:
    """学習済みBugHunterモデルを使用した特徴量分析クラス"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: 学習済みモデルのpklファイルパス
        """
        self.model_path = model_path
        self.model_data = None
        self.load_model()

    def load_model(self):
        """学習済みモデルを読み込む"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイル '{self.model_path}' が見つかりません")

        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        print(f"モデルを '{self.model_path}' から読み込みました")
        self._display_model_info()

    def _display_model_info(self):
        """読み込んだモデルの基本情報を表示"""
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
        """特徴量重要度のテーブルを表示"""
        if not self.model_data['feature_importance_scores'] is not None:
            print("Feature Importanceスコアが見つかりません。")
            return None

        # 全特徴量のDataFrame作成
        all_features_df = pd.DataFrame({
            '特徴量': self.model_data['all_feature_names'],
            'Feature Importance': self.model_data['feature_importance_scores']
        }).sort_values('Feature Importance', ascending=False)

        # 選択された特徴量のみフィルタ
        selected_features_df = all_features_df[
            all_features_df['特徴量'].isin(self.model_data['selected_features'])
        ]

        # 特徴量タイプを追加
        selected_features_df['タイプ'] = selected_features_df['特徴量'].apply(
            lambda x: 'LongName TF-IDF' if x.startswith('LongName_tfidf_')
                      else 'Parent TF-IDF' if x.startswith('Parent_tfidf_')
                      else 'operation_type' if x.startswith('operation_type_')
                      else '数値'
        )

        print(f"\n=== 上位{top_n}特徴量（RandomForest Feature Importance） ===")
        display_df = selected_features_df.head(top_n)[['特徴量', 'タイプ', 'Feature Importance']].copy()
        print(display_df.to_string(index=False))

        # タイプ別統計
        print(f"\n=== 特徴量タイプ別統計（選択された特徴量のみ） ===")
        type_stats = selected_features_df['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_score = selected_features_df[selected_features_df['タイプ'] == feature_type]['Feature Importance'].mean()
            print(f"{feature_type}: {count}個 (平均Feature Importance: {avg_score:.4f})")

        return selected_features_df

    def display_sampling_summary(self):
        """サンプリング情報のサマリーを表示"""
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
        """特徴量選択のサマリーを表示"""
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
        """operation_type特徴量の分析結果を表示"""
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

            # 選択された特徴量の中でoperation_type関連のものをチェック
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
        """交差検証の詳細結果を取得"""
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

    def plot_partial_dependence(self, top_n: int = 20, save_path: Optional[str] = None):
        """特徴量上位N個のPartial Dependence Plotを描画"""

        model = self.model_data['model']
        feature_scores = self.model_data['feature_importance_scores']
        selected_features = self.model_data['selected_features']
        all_features = self.model_data['all_feature_names']

        if not model or feature_scores is None or not selected_features:
            print("PDP描画に必要なデータが不足しています。")
            return

        print(f"\n=== 特徴量上位{top_n}個の分析と可視化 ===")

        # 選択された特徴量の中で重要度上位N個を取得
        selected_features_df = pd.DataFrame({
            '特徴量': all_features,
            'Feature Importance': feature_scores
        })

        # 選択された特徴量のみをフィルタ
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

        # 1. 特徴量重要度チャートを描画
        print("\n=== 1) 特徴量重要度チャート描画 ===")
        importance_save_path = "feature_importance_chart.png" if save_path is None else save_path.replace('.png', '_importance.png')
        self._plot_feature_importance_chart(top_features, feature_scores, all_features, importance_save_path)

        # 2. Partial Dependence Plotsを描画
        print("\n=== 2) Partial Dependence Plots描画 ===")
        pdp_save_path = "partial_dependence_plots.png" if save_path is None else save_path
        self._plot_partial_dependence_plots(top_features, model, selected_features, pdp_save_path)

    def _plot_partial_dependence_plots(self, top_features: List[str], model, selected_features: List[str], save_path: str):
        """実際のPartial Dependence Plotsを描画"""
        try:
            # 訓練データがない場合の代替データ生成
            print("Partial Dependence Plot用のサンプルデータを生成中...")

            # 選択された特徴量の代表的な値範囲を作成
            n_samples = 1000
            sample_data = {}

            for feature in selected_features:
                if feature.startswith('LongName_tfidf_') or feature.startswith('Parent_tfidf_'):
                    # TF-IDF特徴量: 0-1の範囲でランダム値
                    sample_data[feature] = np.random.random(n_samples) * 0.5
                elif feature.startswith('operation_type_'):
                    # One-Hot特徴量: 0と1の両方の値を確実に含める
                    sample_data[feature] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
                else:
                    # 数値特徴量: 標準正規分布
                    sample_data[feature] = np.random.normal(0, 1, n_samples)

            # DataFrameを作成
            X_sample = pd.DataFrame(sample_data)

            # seaborn と matplotlib の設定
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            sns.set(font='IPAexGothic')

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
            for i, feature_name in enumerate(top_features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]

                try:
                    # 特徴量のインデックスを取得
                    feature_idx = selected_features.index(feature_name)

                    # operation_type特徴量の特別処理
                    if feature_name.startswith('operation_type_'):
                        self._plot_operation_type_pdp(ax, feature_name, feature_idx, X_sample, model)
                    else:
                        # 通常のPDP描画
                        try:
                            display = PartialDependenceDisplay.from_estimator(
                                model,
                                X_sample,
                                features=[feature_idx],
                                ax=ax,
                                random_state=42,
                                grid_resolution=20
                            )
                        except Exception as pdp_error:
                            print(f"PDP標準機能でエラー: {pdp_error}")
                            self._plot_manual_pdp_simple(ax, feature_name, feature_idx, X_sample, model)

                    # タイトルを設定（特徴量名を短縮）
                    short_name = self._shorten_feature_name(feature_name)
                    all_features = self.model_data['all_feature_names']
                    feature_scores = self.model_data['feature_importance_scores']
                    importance_idx = all_features.index(feature_name)
                    importance = feature_scores[importance_idx]
                    ax.set_title(f'{short_name}\n(重要度: {importance:.3f})', fontsize=10, pad=10)

                    # 軸ラベルのフォントサイズを調整
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.set_xlabel('特徴量の値', fontsize=9)
                    ax.set_ylabel('Partial Dependence', fontsize=9)

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

            # メインタイトルを追加
            fig.suptitle(f'Partial Dependence Plots (上位{len(top_features)}特徴量)', fontsize=16, y=0.98)

            # レイアウトを調整
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 保存
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Partial Dependence Plotsを '{save_path}' に保存しました")

            plt.show()
            print("Partial Dependence Plots描画完了")

        except Exception as e:
            print(f"PDP描画中にエラーが発生しました: {e}")
            print("代替手法: 特徴量分布ヒストグラムを表示します")
            self._plot_feature_distributions_alternative(top_features, save_path)

    def _plot_operation_type_pdp(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame, model):
        """operation_type特徴量専用のPDP描画"""
        try:
            # operation_type特徴量のユニーク値を確認
            feature_data = X_sample.iloc[:, feature_idx]
            unique_values = sorted(feature_data.unique())

            print(f"  {feature_name} のユニーク値: {unique_values}")

            # 0と1の両方が存在することを確認
            if len(unique_values) < 2 or 0 not in unique_values or 1 not in unique_values:
                # 強制的に0と1の両方を含むデータセットを作成
                print(f"  {feature_name} に0と1の両方を強制追加")
                X_modified = X_sample.copy()

                # 半分を0、半分を1に設定
                n_half = len(X_modified) // 2
                X_modified.iloc[:n_half, feature_idx] = 0
                X_modified.iloc[n_half:, feature_idx] = 1

                feature_data = X_modified.iloc[:, feature_idx]
                unique_values = [0, 1]
            else:
                X_modified = X_sample

            # 手動でPartial Dependenceを計算
            pdp_values = []

            for value in unique_values:
                # 特徴量の値を固定してPDを計算
                X_temp = X_modified.copy()
                X_temp.iloc[:, feature_idx] = value

                # 予測確率を計算
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_value = np.mean(predictions)
                pdp_values.append(pdp_value)

            # 棒グラフで描画
            colors = ['#FF6B6B' if val == 0 else '#4ECDC4' for val in unique_values]
            bars = ax.bar(range(len(unique_values)), pdp_values,
                         color=colors, alpha=0.7, edgecolor='navy', linewidth=1)

            # X軸の設定
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([f'{val}' for val in unique_values])

            # 値をバーの上に表示
            for i, (bar, pdp_val) in enumerate(zip(bars, pdp_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(pdp_values) * 0.01,
                       f'{pdp_val:.3f}', ha='center', va='bottom', fontsize=8)

            # グリッドを追加
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            print(f"  {feature_name} PDP値: {dict(zip(unique_values, pdp_values))}")

        except Exception as e:
            print(f"  operation_type PDP描画でエラー: {e}")
            # 最終的な代替表示
            self._plot_simple_bar_alternative(ax, feature_name)

    def _plot_manual_pdp_simple(self, ax, feature_name: str, feature_idx: int, X_sample: pd.DataFrame, model):
        """シンプルな手動PDP描画"""
        try:
            feature_data = X_sample.iloc[:, feature_idx]

            # 値の範囲を取得
            min_val, max_val = feature_data.min(), feature_data.max()

            # 値の範囲が極端に小さい場合の処理
            if abs(max_val - min_val) < 1e-10:
                self._plot_simple_bar_alternative(ax, feature_name)
                return

            # 適切な値の範囲でPDPを計算
            if len(feature_data.unique()) <= 10:
                # 離散値の場合
                test_values = sorted(feature_data.unique())
            else:
                # 連続値の場合
                test_values = np.linspace(min_val, max_val, 20)

            pdp_values = []
            for value in test_values:
                X_temp = X_sample.copy()
                X_temp.iloc[:, feature_idx] = value
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_values.append(np.mean(predictions))

            # プロット描画
            if len(test_values) <= 10:
                # 離散値: 棒グラフ
                ax.bar(range(len(test_values)), pdp_values, alpha=0.7, color='skyblue', edgecolor='navy')
                ax.set_xticks(range(len(test_values)))
                ax.set_xticklabels([f'{val:.2f}' for val in test_values])
            else:
                # 連続値: 線グラフ
                ax.plot(test_values, pdp_values, 'o-', color='blue', linewidth=2, markersize=4)

            ax.grid(True, alpha=0.3, linestyle='--')

        except Exception as e:
            print(f"  手動PDP描画でエラー: {e}")
            self._plot_simple_bar_alternative(ax, feature_name)

    def _plot_simple_bar_alternative(self, ax, feature_name: str):
        """最も単純な代替表示"""
        # 単純な情報表示
        ax.text(0.5, 0.5, f'{self._shorten_feature_name(feature_name)}\n\nPDP計算困難\n(単一値または\nデータ不足)',
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", alpha=0.8, edgecolor='blue'))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#f8f8f8')

        # Y軸ラベルは統一
        ax.set_ylabel('Partial Dependence', fontsize=9)

    def _plot_error_alternative(self, ax, feature_name: str, error):
        """エラーが発生した場合の代替表示"""
        ax.text(0.5, 0.5, f'描画エラー:\n{self._shorten_feature_name(feature_name)}\n\n{str(error)[:50]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8, edgecolor='red'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Partial Dependence', fontsize=9)
        ax.set_facecolor('#f8f8f8')

    def _plot_feature_distributions_alternative(self, top_features: List[str], save_path: str):
        """PDPが失敗した場合の代替：特徴量重要度の別形式表示"""
        print("代替表示: 特徴量重要度のレーダーチャートを作成します")

        try:
            all_features = self.model_data['all_feature_names']
            feature_scores = self.model_data['feature_importance_scores']

            # 重要度データを取得
            importances = []
            for feature in top_features[:10]:  # 上位10個に限定
                idx = all_features.index(feature)
                importances.append(feature_scores[idx])

            # レーダーチャートの準備
            features_short = [self._shorten_feature_name(f, 15) for f in top_features[:10]]

            # 角度の計算
            angles = np.linspace(0, 2 * np.pi, len(features_short), endpoint=False).tolist()
            importances += importances[:1]  # 閉じるために最初の値を最後に追加
            angles += angles[:1]

            # プロット作成
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # レーダーチャート描画
            ax.plot(angles, importances, 'o-', linewidth=2, color='#4ECDC4')
            ax.fill(angles, importances, alpha=0.25, color='#4ECDC4')

            # ラベル設定
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features_short)
            ax.set_ylim(0, max(importances) * 1.1)
            ax.set_title('特徴量重要度 (上位10特徴量)', size=16, pad=20)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"代替チャートを '{save_path}' に保存しました")
            plt.show()

        except Exception as e:
            print(f"代替表示でもエラーが発生しました: {e}")

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

    def _plot_feature_importance_chart(self, top_features: List[str], feature_scores: np.ndarray,
                                     all_features: List[str], save_path: Optional[str] = None):
        """特徴量重要度のチャートを描画"""
        # seaborn と matplotlib の設定
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set(font='IPAexGothic')

        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

        # 特徴量重要度の値を取得
        importances = []
        feature_types = []
        short_names = []

        for feature in top_features:
            idx = all_features.index(feature)
            importances.append(feature_scores[idx])
            feature_types.append(self._get_feature_type(feature))
            short_names.append(self._shorten_feature_name(feature))

        # カラーマップを特徴量タイプごとに設定
        type_colors = {
            'LongName TF-IDF': '#FF6B6B',
            'Parent TF-IDF': '#4ECDC4',
            'operation_type': '#45B7D1',
            '数値': '#96CEB4'
        }

        colors = [type_colors.get(ft, '#95A5A6') for ft in feature_types]

        # 横棒グラフを作成
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.4)))

        # 特徴量を重要度の降順で並べる（上から重要度が高い順）
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # ラベル設定
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'特徴量重要度 Top {len(top_features)}', fontsize=14, pad=20)

        # 重要度の値をバーの右端に表示
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center', fontsize=9)

        # 凡例を追加
        unique_types = list(set(feature_types))
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=type_colors.get(t, '#95A5A6'),
                                       alpha=0.8, edgecolor='black') for t in unique_types]
        ax.legend(legend_elements, unique_types, loc='lower right', fontsize=10)

        # Y軸を反転（重要度が高い順に上から表示）
        ax.invert_yaxis()

        # グリッドの調整
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        # レイアウト調整
        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"特徴量重要度チャートを '{save_path}' に保存しました")

        plt.show()
        print("特徴量重要度チャート描画完了")

    def _shorten_feature_name(self, feature_name: str, max_length: int = 25) -> str:
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
            return f"op_{op_type[:15]}"
        else:
            # 数値特徴量は前後を切り詰め
            if len(feature_name) > max_length:
                return feature_name[:max_length-3] + "..."
            return feature_name

    def plot_cv_results(self, save_path: Optional[str] = None):
        """交差検証結果の可視化"""
        cv_results = self.model_data.get('cv_results')
        if not cv_results:
            print("交差検証結果が見つかりません。")
            return

        # データの準備
        metrics = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']
        metric_names = ['F1スコア', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # seaborn設定
        sns.set_style("whitegrid")
        sns.set(font='IPAexGothic')

        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            scores = cv_results[f'{metric}_scores']

            # ボックスプロット
            bp = ax.boxplot([scores], patch_artist=True, labels=[metric_name])
            bp['boxes'][0].set_facecolor('#4ECDC4')
            bp['boxes'][0].set_alpha(0.7)

            # 平均値と標準偏差を表示
            mean_val = cv_results[f'{metric}_mean']
            std_val = cv_results[f'{metric}_std']

            ax.text(0.5, 0.95, f'平均: {mean_val:.4f}\n標準偏差: {std_val:.4f}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

            ax.set_title(f'{metric_name} (10分割交差検証)', fontsize=12)
            ax.grid(True, alpha=0.3)

        # 6番目のサブプロットは非表示
        axes[5].set_visible(False)

        plt.suptitle('交差検証結果サマリー', fontsize=16, y=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"交差検証結果チャートを '{save_path}' に保存しました")

        plt.show()

    def generate_analysis_report(self):
        """包括的な分析レポートを生成"""
        print("="*80)
        print("BugHunter モデル分析レポート")
        print("="*80)

        # モデル基本情報
        self._display_model_info()

        # サンプリング情報
        self.display_sampling_summary()

        # 特徴量選択情報
        self.display_feature_selection_summary()

        # operation_type分析
        self.display_operation_type_analysis()

        # 特徴量重要度
        feature_df = self.display_feature_importance_table(top_n=20)

        # 交差検証詳細結果
        cv_df = self.get_cv_detailed_results()
        if cv_df is not None:
            print("\n=== 各フォールドの詳細結果（交差検証） ===")
            print(cv_df.round(4))

        print("\n" + "="*80)
        print("分析レポート完了")
        print("="*80)


def main():
    """分析の実行例"""
    try:
        # 学習済みモデルを読み込んで分析
        analyzer = BugHunterAnalyzer("predictions_nan.pkl")

        # 包括的な分析レポートを生成
        analyzer.generate_analysis_report()

        # 特徴量重要度チャートを描画
        analyzer.plot_partial_dependence(
            top_n=20,
            save_path="analysis_charts.png"  # feature_importance_chart.png と partial_dependence_plots.png が生成される
        )

        # 交差検証結果を可視化
        analyzer.plot_cv_results(save_path="cv_results_chart.png")

        print("\n分析完了！")

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("まず bug_hunter_trainer.py を実行してモデルを訓練してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
