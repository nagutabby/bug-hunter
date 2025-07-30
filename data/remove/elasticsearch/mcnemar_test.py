"""
マクネマー検定による予測モデル性能比較（簡潔版）
"""

import numpy as np
import pickle
import os
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from trainer import JavaCodeTokenizer

warnings.filterwarnings('ignore')


class McNemarTest:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def _extract_predictions(self, model_data):
        if hasattr(model_data, 'predictions_data') and model_data.predictions_data is not None:
            return model_data.predictions_data
        elif isinstance(model_data, dict) and 'predictions_data' in model_data:
            return model_data['predictions_data']
        elif isinstance(model_data, dict) and all(key in model_data for key in ['y_true', 'y_pred']):
            return model_data
        else:
            raise ValueError("予測結果の形式が不正です")

    def load_predictions(self, file_path_1, file_path_2):
        with open(file_path_1, 'rb') as f:
            model_data_1 = pickle.load(f)
        with open(file_path_2, 'rb') as f:
            model_data_2 = pickle.load(f)

        self.pred_1 = self._extract_predictions(model_data_1)
        self.pred_2 = self._extract_predictions(model_data_2)

        if len(self.pred_1['y_true']) != len(self.pred_2['y_true']):
            raise ValueError("データサイズが異なります")
        if not np.array_equal(self.pred_1['y_true'], self.pred_2['y_true']):
            raise ValueError("正解ラベルが一致しません")

    def create_contingency_table(self):
        y_true = self.pred_1['y_true']
        correct_1 = (self.pred_1['y_pred'] == y_true)
        correct_2 = (self.pred_2['y_pred'] == y_true)

        both_correct = np.sum(correct_1 & correct_2)
        model1_only = np.sum(correct_1 & ~correct_2)
        model2_only = np.sum(~correct_1 & correct_2)
        both_wrong = np.sum(~correct_1 & ~correct_2)

        return np.array([[both_correct, model1_only], [model2_only, both_wrong]])

    def perform_test(self):
        table = self.create_contingency_table()

        b = table[0, 1]
        c = table[1, 0]
        total_disagreements = b + c

        if total_disagreements < 25:
            result = mcnemar(table, exact=True)
            test_type = "正確検定"
        else:
            result = mcnemar(table, exact=False, correction=True)
            test_type = "カイ二乗検定"

        return {
            'p_value': result.pvalue,
            'is_significant': result.pvalue < self.alpha,
            'test_type': test_type,
            'model1_only_correct': b,
            'model2_only_correct': c,
            'total_disagreements': total_disagreements
        }

    def compare_models(self, file_path_1, file_path_2, model_name_1="モデル1", model_name_2="モデル2"):
        self.load_predictions(file_path_1, file_path_2)
        result = self.perform_test()

        print(f"=== マクネマー検定結果 ===")
        print(f"比較対象: {model_name_1} vs {model_name_2}")
        print(f"使用検定: {result['test_type']}")
        print(f"p値: {result['p_value']:.6f}")
        print(f"有意水準: {self.alpha}")
        print(f"統計的有意差: {'あり' if result['is_significant'] else 'なし'}")

        if result['is_significant']:
            if result['model2_only_correct'] > result['model1_only_correct']:
                print(f"結論: {model_name_2}が{model_name_1}より有意に優れています")
            else:
                print(f"結論: {model_name_1}が{model_name_2}より有意に優れています")
        else:
            print(f"結論: 両モデルの性能に有意差はありません")

        return result


def main():
    test = McNemarTest(alpha=0.05)

    file_1 = "predictions_current.pkl"
    file_2 = "predictions_changes.pkl"

    if not os.path.exists(file_1) or not os.path.exists(file_2):
        print("予測結果ファイルが見つかりません")
        return

    try:
        result = test.compare_models(file_1, file_2)
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()
