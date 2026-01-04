import numpy as np
import pickle
import os
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from train import JavaCodeTokenizer

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

    def compare_models(self, file_path_1, file_path_2, model_name_1="Base", model_name_2="Improved"):
        self.load_predictions(file_path_1, file_path_2)
        result = self.perform_test()

        print(f"\n使用検定: {result['test_type']}")
        print(f"不一致数: {result['total_disagreements']} ({model_name_1}のみ正解: {result['model1_only_correct']}, {model_name_2}のみ正解: {result['model2_only_correct']})")
        print(f"p値: {result['p_value']:.6f}")
        print(f"有意水準α={self.alpha}: {'有意' if result['is_significant'] else '有意差なし'}")

        if result['is_significant']:
            if result['model2_only_correct'] > result['model1_only_correct']:
                print(f"→ {model_name_2}が{model_name_1}より有意に優れている")
            else:
                print(f"→ {model_name_1}が{model_name_2}より有意に優れている")

        return result

def analyze_project(test, project_name, base_model_path, improved_model_path):
    print(f"\n{'='*60}")
    print(f"プロジェクト: {project_name}")
    print('='*60)

    if not os.path.exists(base_model_path) or not os.path.exists(improved_model_path):
        print("予測結果ファイルが見つかりません")
        return None

    try:
        result = test.compare_models(base_model_path, improved_model_path,
                                     model_name_1="Base", model_name_2="Improved")
        result['project'] = project_name
        return result
    except Exception as e:
        print(f"エラー: {e}")
        return None

def main():
    projects = [
        {
            'name': 'elasticsearch',
            'base_model': '../data/remove/elasticsearch/predictions_base.pkl',
            'improved_model': '../data/remove/elasticsearch/predictions_add_method_commit_level_metrics.pkl'
        },
        {
            'name': 'hazelcast',
            'base_model': '../data/remove/hazelcast/predictions_base.pkl',
            'improved_model': '../data/remove/hazelcast/predictions_add_method_commit_level_metrics.pkl'
        },
        {
            'name': 'neo4j',
            'base_model': '../data/remove/neo4j/predictions_base.pkl',
            'improved_model': '../data/remove/neo4j/predictions_add_method_commit_level_metrics.pkl'
        },
        {
            'name': 'netty',
            'base_model': '../data/remove/netty/predictions_base.pkl',
            'improved_model': '../data/remove/netty/predictions_add_method_commit_level_metrics.pkl'
        },
        {
            'name': 'orientdb',
            'base_model': '../data/remove/orientdb/predictions_base.pkl',
            'improved_model': '../data/remove/orientdb/predictions_add_method_commit_level_metrics.pkl'
        }
    ]

    test = McNemarTest(alpha=0.05)

    print("=" * 80)
    print("全プロジェクトに対するマクネマー検定（有意水準α=0.05）")
    print("=" * 80)

    results = []
    for project in projects:
        result = analyze_project(
            test=test,
            project_name=project['name'],
            base_model_path=project['base_model'],
            improved_model_path=project['improved_model']
        )
        if result is not None:
            results.append(result)

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results[['project', 'test_type', 'p_value', 'is_significant',
                                 'model1_only_correct', 'model2_only_correct', 'total_disagreements']]

if __name__ == "__main__":
    main()
