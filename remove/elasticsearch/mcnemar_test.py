#!/usr/bin/env python3
"""
マクネマー検定による予測モデル性能比較プログラム

このプログラムは、lizard_analysis_nan.pyとlizard_analysis_real_number.pyによって
作成された2つの異なるデータセットを使用して学習したRandomForestモデルの
予測結果に有意差があるかをマクネマー検定で確認します。

使用方法:
1. random_forest.pyで各データセットを使って学習・予測を実行
2. 予測結果をpickleファイルで保存
3. このプログラムで2つの予測結果を比較

有意水準: 5% (p = 0.05)
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import warnings
from typing import Tuple, Dict, Optional
import statsmodels.stats.contingency_tables as ct
from statsmodels.stats.contingency_tables import mcnemar
from trainer import JavaCodeTokenizer
warnings.filterwarnings('ignore')

class McNemarTestAnalyzer:
    """マクネマー検定による予測結果比較アナライザー"""

    def __init__(self, alpha: float = 0.05):
        """
        初期化

        Parameters:
        -----------
        alpha : float
            有意水準（デフォルト: 0.05）
        """
        self.alpha = alpha
        self.predictions_1 = None
        self.predictions_2 = None
        self.test_results = None

    def _extract_predictions_from_model(self, model_data, model_name: str) -> dict:
        """
        保存されたモデルデータから予測結果を抽出

        Parameters:
        -----------
        model_data : dict or object
            pickleファイルから読み込んだデータ
        model_name : str
            モデル名（エラーメッセージ用）

        Returns:
        --------
        dict
            予測結果辞書
        """
        # trainer.pyで保存される形式に対応
        if hasattr(model_data, 'predictions_data') and model_data.predictions_data is not None:
            return model_data.predictions_data

        # 直接predictions_dataが保存されている場合
        elif isinstance(model_data, dict) and 'predictions_data' in model_data:
            return model_data['predictions_data']

        # 直接予測結果が保存されている場合
        elif isinstance(model_data, dict) and all(key in model_data for key in ['y_true', 'y_pred']):
            return model_data

        # trainer オブジェクト全体が保存されている場合の詳細チェック
        elif hasattr(model_data, '__dict__'):
            # オブジェクトの属性を確認
            attrs = dir(model_data)
            print(f"  {model_name} オブジェクトの主な属性: {[attr for attr in attrs if not attr.startswith('_')]}")

            # predictions_data属性があるか確認
            if hasattr(model_data, 'predictions_data'):
                pred_data = model_data.predictions_data
                if pred_data is not None:
                    return pred_data
                else:
                    raise ValueError(f"{model_name}: predictions_dataがNoneです。モデルの学習とテスト評価が完了していない可能性があります。")
            else:
                raise ValueError(f"{model_name}: predictions_data属性が見つかりません。")

        else:
            raise ValueError(f"{model_name}: 予測結果の形式が不明です。型: {type(model_data)}")

    def load_predictions(self, file_path_1: str, file_path_2: str,
                        model_name_1: str = "Model 1", model_name_2: str = "Model 2"):
        """
        2つの予測結果ファイルを読み込む

        Parameters:
        -----------
        file_path_1 : str
            1つ目の予測結果ファイルパス（例: NaN版の結果）
        file_path_2 : str
            2つ目の予測結果ファイルパス（例: 実数版の結果）
        model_name_1 : str
            1つ目のモデルの名前
        model_name_2 : str
            2つ目のモデルの名前
        """
        # ファイル存在確認
        if not os.path.exists(file_path_1):
            raise FileNotFoundError(f"ファイル '{file_path_1}' が見つかりません")
        if not os.path.exists(file_path_2):
            raise FileNotFoundError(f"ファイル '{file_path_2}' が見つかりません")

        # 予測結果読み込み
        print(f"=== 予測結果の読み込み ===")
        print(f"{model_name_1}: {file_path_1}")

        try:
            with open(file_path_1, 'rb') as f:
                model_data_1 = pickle.load(f)

            self.predictions_1 = self._extract_predictions_from_model(model_data_1, model_name_1)
            print(f"  読み込み完了: {len(self.predictions_1['y_true'])}件のテストデータ")

        except Exception as e:
            print(f"  エラー詳細: {e}")
            raise ValueError(f"{model_name_1}の予測結果の読み込みに失敗しました: {e}")

        print(f"{model_name_2}: {file_path_2}")

        try:
            with open(file_path_2, 'rb') as f:
                model_data_2 = pickle.load(f)

            self.predictions_2 = self._extract_predictions_from_model(model_data_2, model_name_2)
            print(f"  読み込み完了: {len(self.predictions_2['y_true'])}件のテストデータ")

        except Exception as e:
            print(f"  エラー詳細: {e}")
            raise ValueError(f"{model_name_2}の予測結果の読み込みに失敗しました: {e}")

        # データ整合性確認
        self._validate_predictions(model_name_1, model_name_2)

        # モデル名を保存
        self.model_name_1 = model_name_1
        self.model_name_2 = model_name_2

        print(f"予測結果の読み込みが完了しました")

    def _validate_predictions(self, model_name_1: str, model_name_2: str):
        """予測結果の整合性を確認"""

        # 必要なキーの存在確認
        required_keys = ['y_true', 'y_pred']
        for predictions, model_name in [(self.predictions_1, model_name_1), (self.predictions_2, model_name_2)]:
            for key in required_keys:
                if key not in predictions:
                    raise ValueError(f"{model_name}: 必要なキー '{key}' が見つかりません。利用可能なキー: {list(predictions.keys())}")

        # データサイズの確認
        if len(self.predictions_1['y_true']) != len(self.predictions_2['y_true']):
            raise ValueError(f"テストデータのサイズが異なります: "
                           f"{model_name_1}={len(self.predictions_1['y_true'])}, "
                           f"{model_name_2}={len(self.predictions_2['y_true'])}")

        # 正解ラベルの一致確認
        if not np.array_equal(self.predictions_1['y_true'], self.predictions_2['y_true']):
            raise ValueError("2つのモデルの正解ラベル（y_true）が一致しません。"
                           "同一のテストデータセットを使用してください。")

        # テストインデックスの確認（存在する場合）
        if ('test_indices' in self.predictions_1 and 'test_indices' in self.predictions_2):
            if self.predictions_1['test_indices'] != self.predictions_2['test_indices']:
                print("警告: テストデータのインデックスが異なりますが、処理を続行します")

        print("✓ データ整合性確認完了")

    def create_contingency_table(self) -> pd.DataFrame:
        """
        2×2分割表を作成

        Returns:
        --------
        pd.DataFrame
            分割表（contingency table）
        """
        if self.predictions_1 is None or self.predictions_2 is None:
            raise ValueError("予測結果が読み込まれていません。load_predictions()を先に実行してください。")

        y_true = self.predictions_1['y_true']
        y_pred_1 = self.predictions_1['y_pred']
        y_pred_2 = self.predictions_2['y_pred']

        # 各モデルの正解/不正解を計算
        correct_1 = (y_pred_1 == y_true)
        correct_2 = (y_pred_2 == y_true)

        # 2×2分割表の各セルを計算
        # both_correct: 両方とも正解
        # model1_only: モデル1のみ正解
        # model2_only: モデル2のみ正解
        # both_wrong: 両方とも不正解

        both_correct = np.sum(correct_1 & correct_2)
        model1_only = np.sum(correct_1 & ~correct_2)
        model2_only = np.sum(~correct_1 & correct_2)
        both_wrong = np.sum(~correct_1 & ~correct_2)

        # 分割表作成
        contingency_table = pd.DataFrame({
            f'{self.model_name_2} 正解': [both_correct, model2_only],
            f'{self.model_name_2} 不正解': [model1_only, both_wrong]
        }, index=[f'{self.model_name_1} 正解', f'{self.model_name_1} 不正解'])

        return contingency_table

    def calculate_basic_metrics(self) -> Dict:
        """基本的な性能指標を計算"""

        y_true = self.predictions_1['y_true']
        y_pred_1 = self.predictions_1['y_pred']
        y_pred_2 = self.predictions_2['y_pred']

        # 各モデルの精度計算
        accuracy_1 = np.mean(y_pred_1 == y_true)
        accuracy_2 = np.mean(y_pred_2 == y_true)

        # 予測の不一致数
        disagreement = np.sum(y_pred_1 != y_pred_2)
        total_samples = len(y_true)

        # 各クラスの分布
        class_distribution = {
            'class_0': np.sum(y_true == 0),
            'class_1': np.sum(y_true == 1)
        }

        return {
            'total_samples': total_samples,
            'accuracy_1': accuracy_1,
            'accuracy_2': accuracy_2,
            'accuracy_difference': accuracy_2 - accuracy_1,
            'disagreement_count': disagreement,
            'disagreement_rate': disagreement / total_samples,
            'class_distribution': class_distribution
        }

    def perform_mcnemar_test(self) -> Dict:
        """
        マクネマー検定を実行

        Returns:
        --------
        Dict
            検定結果
        """
        if self.predictions_1 is None or self.predictions_2 is None:
            raise ValueError("予測結果が読み込まれていません。load_predictions()を先に実行してください。")

        # 分割表作成
        contingency_table = self.create_contingency_table()

        # マクネマー検定に必要な値を抽出
        # contingency_table.iloc[0,1] = モデル1のみ正解 (b)
        # contingency_table.iloc[1,0] = モデル2のみ正解 (c)
        b = contingency_table.iloc[0, 1]  # モデル1のみ正解
        c = contingency_table.iloc[1, 0]  # モデル2のみ正解

        print(f"\n=== マクネマー検定のための値 ===")
        print(f"b ({self.model_name_1}のみ正解): {b}")
        print(f"c ({self.model_name_2}のみ正解): {c}")
        print(f"b + c (不一致総数): {b + c}")

        # マクネマー検定実行
        # exact=Falseにして連続性補正ありのカイ二乗検定を使用
        # exact=Trueにすると二項検定になる

        # 連続性補正ありカイ二乗検定
        result_chi2 = mcnemar(contingency_table.values, exact=False, correction=True)

        # 連続性補正なしカイ二乗検定
        result_chi2_no_corr = mcnemar(contingency_table.values, exact=False, correction=False)

        # 正確な二項検定（サンプルサイズが小さい場合に適用）
        result_exact = mcnemar(contingency_table.values, exact=True)

        # 効果量計算（オッズ比）
        odds_ratio = b / c if c > 0 else float('inf')

        # 結果をまとめる
        test_results = {
            'contingency_table': contingency_table,
            'b_model1_only_correct': b,
            'c_model2_only_correct': c,
            'total_disagreements': b + c,

            # カイ二乗検定（連続性補正あり）
            'chi2_corrected': {
                'statistic': result_chi2.statistic,
                'pvalue': result_chi2.pvalue,
                'is_significant': result_chi2.pvalue < self.alpha
            },

            # カイ二乗検定（連続性補正なし）
            'chi2_no_correction': {
                'statistic': result_chi2_no_corr.statistic,
                'pvalue': result_chi2_no_corr.pvalue,
                'is_significant': result_chi2_no_corr.pvalue < self.alpha
            },

            # 正確検定
            'exact_test': {
                'statistic': result_exact.statistic,
                'pvalue': result_exact.pvalue,
                'is_significant': result_exact.pvalue < self.alpha
            },

            # 効果量
            'odds_ratio': odds_ratio,
            'alpha': self.alpha
        }

        self.test_results = test_results
        return test_results

    def interpret_results(self) -> str:
        """検定結果の解釈を生成"""

        if self.test_results is None:
            raise ValueError("検定が実行されていません。perform_mcnemar_test()を先に実行してください。")

        b = self.test_results['b_model1_only_correct']
        c = self.test_results['c_model2_only_correct']

        chi2_corrected = self.test_results['chi2_corrected']
        exact_test = self.test_results['exact_test']

        # 推奨する検定方法を決定
        total_disagreements = b + c
        if total_disagreements < 25:
            recommended_test = 'exact_test'
            test_name = "正確な二項検定"
            p_value = exact_test['pvalue']
            is_significant = exact_test['is_significant']
        else:
            recommended_test = 'chi2_corrected'
            test_name = "カイ二乗検定（連続性補正あり）"
            p_value = chi2_corrected['pvalue']
            is_significant = chi2_corrected['is_significant']

        # 解釈文作成
        interpretation = f"""
=== マクネマー検定結果の解釈 ===

1. 検定の前提条件:
   - 不一致総数 (b + c): {total_disagreements}
   - 推奨検定方法: {test_name}
   - 理由: {'サンプルサイズが小さいため' if total_disagreements < 25 else 'サンプルサイズが十分'}

2. 検定結果:
   - p値: {p_value:.6f}
   - 有意水準: {self.alpha}
   - 統計的有意差: {'あり' if is_significant else 'なし'}

3. 結論:
"""

        if is_significant:
            # 有意差がある場合
            if c > b:
                better_model = self.model_name_2
                worse_model = self.model_name_1
            else:
                better_model = self.model_name_1
                worse_model = self.model_name_2

            interpretation += f"""   有意水準{self.alpha}で、2つのモデルの予測性能に統計的有意差があります。
   {better_model}が{worse_model}よりも有意に優れた性能を示しています。

   実践的含意:
   - lizard_analysis_nan.pyとlizard_analysis_real_number.pyによって作成される
     特徴量の違いが、RandomForestモデルの予測性能に有意な影響を与えています。
   - より優れた性能を示した{better_model}の手法を採用することを推奨します。"""
        else:
            # 有意差がない場合
            interpretation += f"""   有意水準{self.alpha}で、2つのモデルの予測性能に統計的有意差は認められません。

   実践的含意:
   - lizard_analysis_nan.pyとlizard_analysis_real_number.pyによって作成される
     特徴量の違いは、RandomForestモデルの予測性能に統計的に有意な影響を与えていません。
   - どちらの手法を選択しても、統計的には同等の性能が期待できます。
   - その他の基準（計算効率、解釈しやすさ等）で手法を選択することができます。"""

        return interpretation

    def generate_detailed_report(self) -> str:
        """詳細なレポートを生成"""

        if self.test_results is None:
            raise ValueError("検定が実行されていません。perform_mcnemar_test()を先に実行してください。")

        # 基本指標計算
        basic_metrics = self.calculate_basic_metrics()

        # レポート作成
        report = f"""
{'='*80}
マクネマー検定による予測モデル性能比較レポート
{'='*80}

1. 比較対象:
   - {self.model_name_1}: NaN値での operation_type 処理
   - {self.model_name_2}: 実数値での operation_type 処理

2. データ概要:
   - テストサンプル数: {basic_metrics['total_samples']:,}
   - クラス 0: {basic_metrics['class_distribution']['class_0']:,}件
   - クラス 1: {basic_metrics['class_distribution']['class_1']:,}件

3. 基本性能指標:
   - {self.model_name_1}の精度: {basic_metrics['accuracy_1']:.4f}
   - {self.model_name_2}の精度: {basic_metrics['accuracy_2']:.4f}
   - 精度差 ({self.model_name_2} - {self.model_name_1}): {basic_metrics['accuracy_difference']:+.4f}
   - 予測不一致数: {basic_metrics['disagreement_count']:,}件 ({basic_metrics['disagreement_rate']:.4f})

4. 分割表 (Contingency Table):
{self.test_results['contingency_table'].to_string()}

5. マクネマー検定結果:

   a) カイ二乗検定（連続性補正あり）:
      - 統計量: {self.test_results['chi2_corrected']['statistic']:.6f}
      - p値: {self.test_results['chi2_corrected']['pvalue']:.6f}
      - 有意差: {'あり' if self.test_results['chi2_corrected']['is_significant'] else 'なし'}

   b) カイ二乗検定（連続性補正なし）:
      - 統計量: {self.test_results['chi2_no_correction']['statistic']:.6f}
      - p値: {self.test_results['chi2_no_correction']['pvalue']:.6f}
      - 有意差: {'あり' if self.test_results['chi2_no_correction']['is_significant'] else 'なし'}

   c) 正確な二項検定:
      - 統計量: {self.test_results['exact_test']['statistic']:.6f}
      - p値: {self.test_results['exact_test']['pvalue']:.6f}
      - 有意差: {'あり' if self.test_results['exact_test']['is_significant'] else 'なし'}

6. 効果量:
   - オッズ比: {self.test_results['odds_ratio']:.4f}

{self.interpret_results()}

{'='*80}
"""
        return report

    def save_report(self, file_path: str):
        """レポートをファイルに保存"""
        report = self.generate_detailed_report()

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nレポートを '{file_path}' に保存しました")

    def display_results(self):
        """結果を画面に表示"""
        print(self.generate_detailed_report())


def main():
    """メイン関数 - 使用例"""

    print("=== マクネマー検定による予測モデル性能比較 ===")
    print("このプログラムは、異なる特徴量処理方法で学習したRandomForestモデルの")
    print("予測性能に統計的有意差があるかを検定します。")
    print()

    # 予測結果ファイルのパス設定
    predictions_nan_path = "predictions_nan.pkl"
    predictions_real_path = "predictions_real_number.pkl"

    # ファイル存在確認
    if not os.path.exists(predictions_nan_path):
        print(f"エラー: ファイル '{predictions_nan_path}' が見つかりません。")
        print("先にtrainer.pyでlizard_analysis_nan.pyの結果を使って学習・予測を実行してください。")
        return

    if not os.path.exists(predictions_real_path):
        print(f"エラー: ファイル '{predictions_real_path}' が見つかりません。")
        print("先にtrainer.pyでlizard_analysis_real_number.pyの結果を使って学習・予測を実行してください。")
        return

    try:
        # アナライザー初期化
        analyzer = McNemarTestAnalyzer(alpha=0.05)

        # 予測結果読み込み
        analyzer.load_predictions(
            file_path_1=predictions_nan_path,
            file_path_2=predictions_real_path,
            model_name_1="NaN版モデル",
            model_name_2="実数版モデル"
        )

        # マクネマー検定実行
        print("\n=== マクネマー検定実行中... ===")
        test_results = analyzer.perform_mcnemar_test()

        # 結果表示
        analyzer.display_results()

        # レポート保存
        report_path = "mcnemar_test_report.txt"
        analyzer.save_report(report_path)

        # 追加分析: 分割表の詳細
        print("\n=== 追加分析: 分割表の詳細 ===")
        contingency_table = test_results['contingency_table']

        total = contingency_table.values.sum()
        print(f"分割表の各セルの割合:")
        for i, row_name in enumerate(contingency_table.index):
            for j, col_name in enumerate(contingency_table.columns):
                count = contingency_table.iloc[i, j]
                percentage = count / total * 100
                print(f"  {row_name} & {col_name}: {count}件 ({percentage:.2f}%)")

        print(f"\n処理が正常に完了しました。")
        print(f"詳細レポート: {report_path}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


# データ準備用のヘルパー関数
def prepare_comparison_data():
    """
    比較用データの準備手順を表示
    """
    print("""
=== データ準備手順 ===

1. lizard_analysis_nan.pyを実行してoperation_typeカラムにNaN値を含むデータセットを作成:
   python lizard_analysis_nan.py
   → 出力: method-p_filtered_v2_changes_nan.csv

2. lizard_analysis_real_number.pyを実行して実数値のデータセットを作成:
   python lizard_analysis_real_number.py
   → 出力: method-p_filtered_v2_changes_real_number.csv

3. trainer.pyを修正して、各データセットで学習・予測を実行:

   a) NaN版での実行:
      data_path = "method-p_filtered_v2_changes_nan.csv"
      trainer.run_training_pipeline(data_path, max_rows=3000, model_save_path="predictions_nan.pkl")

   b) 実数版での実行:
      data_path = "method-p_filtered_v2_changes_real_number.csv"
      trainer.run_training_pipeline(data_path, max_rows=3000, model_save_path="predictions_real_number.pkl")

4. マクネマー検定実行:
   python mcnemar_test.py

注意: 同一のrandom_seedを使用してテストデータの分割を統一してください。
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        prepare_comparison_data()
    else:
        main()
