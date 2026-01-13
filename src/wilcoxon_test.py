import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from train import JavaCodeTokenizer

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def calculate_effort(code_churn, num_files, entropy):
    if num_files == 0 or code_churn == 0:
        raw_effort = 1.0
    elif num_files == 1:
        raw_effort = float(code_churn)
    elif entropy is None or entropy == 0 or np.isnan(entropy):
        raw_effort = float(code_churn)
    else:
        raw_effort = code_churn * (num_files ** entropy)

    effort = np.log(raw_effort + 1)
    return effort

def knapsack_greedy(weights, values, capacity):
    n = len(weights)

    items = []
    for i in range(n):
        if weights[i] > 0:
            ratio = values[i] / weights[i]
            items.append((i, ratio, weights[i], values[i]))
        else:
            items.append((i, 0, weights[i], values[i]))

    items.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []
    total_weight = 0.0
    total_value = 0.0

    for idx, ratio, weight, value in items:
        if total_weight + weight <= capacity:
            selected_indices.append(idx)
            total_weight += weight
            total_value += value

    selected_indices.sort()
    return selected_indices, total_value, total_weight

def calculate_cost_benefit_at_effort_levels(y_true, y_pred_proba, efforts, effort_levels):
    sorted_efforts = np.sort(efforts)
    lower_80_percent_count = int(len(sorted_efforts) * 0.8)
    total_effort_lower_80 = np.sum(sorted_efforts[:lower_80_percent_count])

    total_bugs = np.sum(y_true)
    results = {}

    for effort_level in effort_levels:
        capacity = total_effort_lower_80 * effort_level

        selected_indices, total_value, total_weight = knapsack_greedy(
            weights=efforts,
            values=y_pred_proba,
            capacity=capacity
        )

        bugs_found = np.sum(y_true[selected_indices])
        bug_detection_ratio = bugs_found / total_bugs if total_bugs > 0 else 0

        results[effort_level] = bug_detection_ratio

    return results

def analyze_project(project_name, base_model_path, improved_model_path,
                   commit_metrics_path, effort_levels):
    print(f"\n{'='*60}")
    print(f"プロジェクト: {project_name}")
    print('='*60)

    model_base = load_model(base_model_path)
    model_improved = load_model(improved_model_path)

    y_true_base = model_base['predictions_data']['y_true']
    y_pred_proba_base = model_base['predictions_data']['y_pred_proba']

    y_true_improved = model_improved['predictions_data']['y_true']
    y_pred_proba_improved = model_improved['predictions_data']['y_pred_proba']

    df_commits = pd.read_csv(commit_metrics_path)

    test_indices = model_improved['predictions_data'].get('test_indices', None)
    if test_indices is not None:
        df_test = df_commits.iloc[test_indices].reset_index(drop=True)
    else:
        df_test = df_commits

    efforts = []
    for idx, row in df_test.iterrows():
        lines_added = row.get('lines_added', 0)
        lines_deleted = row.get('lines_deleted', 0)
        code_churn = lines_added + lines_deleted
        num_files = row.get('num_files', 1)
        entropy = row.get('entropy', 0)
        effort = calculate_effort(code_churn, num_files, entropy)
        efforts.append(effort)

    efforts = np.array(efforts)

    base_results = calculate_cost_benefit_at_effort_levels(
        y_true_base, y_pred_proba_base, efforts, effort_levels
    )
    improved_results = calculate_cost_benefit_at_effort_levels(
        y_true_improved, y_pred_proba_improved, efforts, effort_levels
    )

    print(f"\nバグ総数: {np.sum(y_true_improved)}")
    print("\n各労力レベルでの欠陥発見率:")
    for effort_level in effort_levels:
        base_rate = base_results[effort_level] * 100
        improved_rate = improved_results[effort_level] * 100
        improvement = improved_rate - base_rate
        print(f"  {effort_level*100:.0f}%労力: Base={base_rate:.1f}%, "
              f"Improved={improved_rate:.1f}%, 改善={improvement:+.1f}%ポイント")

    return base_results, improved_results

def perform_wilcoxon_test(all_base_results, all_improved_results, effort_levels, alpha=0.01):
    print(f"\n{'='*80}")
    print("Wilcoxonの符号順位検定（有意水準α={:.2f}）".format(alpha))
    print('='*80)

    n_projects = len(all_base_results)
    print(f"\nプロジェクト数: {n_projects}")

    test_results = []

    for effort_level in effort_levels:
        print(f"\n--- レビュー労力 {effort_level*100:.0f}% での検定 ---")

        base_rates = [results[effort_level] * 100 for results in all_base_results]
        improved_rates = [results[effort_level] * 100 for results in all_improved_results]

        print("\n各プロジェクトの欠陥発見率:")
        for i, (base, improved) in enumerate(zip(base_rates, improved_rates), 1):
            diff = improved - base
            print(f"  プロジェクト{i}: Base={base:.1f}%, Improved={improved:.1f}%, 差={diff:+.1f}%")

        statistic, p_value = wilcoxon(improved_rates, base_rates, alternative='greater')

        is_significant = p_value < alpha
        significance_mark = "***" if is_significant else "n.s."

        print(f"\n検定統計量: {statistic:.4f}")
        print(f"p値: {p_value:.6f}")
        print(f"有意性（α={alpha}）: {significance_mark}")

        if is_significant:
            print(f"→ 提案手法は有意水準{alpha}で統計的に有意に優れている")
        else:
            print(f"→ 有意水準{alpha}では統計的に有意な差は認められない")

        mean_base = np.mean(base_rates)
        mean_improved = np.mean(improved_rates)
        mean_improvement = mean_improved - mean_base

        print(f"\n平均欠陥発見率: Base={mean_base:.1f}%, Improved={mean_improved:.1f}%")
        print(f"平均改善幅: {mean_improvement:+.1f}%ポイント")

        test_results.append({
            'effort_level': f"{effort_level*100:.0f}%",
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'mean_base': mean_base,
            'mean_improved': mean_improved,
            'mean_improvement': mean_improvement
        })

    return pd.DataFrame(test_results)

def main():
    projects = [
        {
            'name': 'elasticsearch',
            'base_model': '../data/remove/elasticsearch/predictions_base.pkl',
            'improved_model': '../data/remove/elasticsearch/predictions_add_method_commit_level_metrics.pkl',
            'commit_metrics': '../data/remove/elasticsearch/method-p_add_method_commit_level_metrics.csv'
        },
        {
            'name': 'hazelcast',
            'base_model': '../data/remove/hazelcast/predictions_base.pkl',
            'improved_model': '../data/remove/hazelcast/predictions_add_method_commit_level_metrics.pkl',
            'commit_metrics': '../data/remove/hazelcast/method-p_add_method_commit_level_metrics.csv'
        },
        {
            'name': 'neo4j',
            'base_model': '../data/remove/neo4j/predictions_base.pkl',
            'improved_model': '../data/remove/neo4j/predictions_add_method_commit_level_metrics.pkl',
            'commit_metrics': '../data/remove/neo4j/method-p_add_method_commit_level_metrics.csv'
        },
        {
            'name': 'netty',
            'base_model': '../data/remove/netty/predictions_base.pkl',
            'improved_model': '../data/remove/netty/predictions_add_method_commit_level_metrics.pkl',
            'commit_metrics': '../data/remove/netty/method-p_add_method_commit_level_metrics.csv'
        },
        {
            'name': 'orientdb',
            'base_model': '../data/remove/orientdb/predictions_base.pkl',
            'improved_model': '../data/remove/orientdb/predictions_add_method_commit_level_metrics.pkl',
            'commit_metrics': '../data/remove/orientdb/method-p_add_method_commit_level_metrics.csv'
        }
    ]

    effort_levels = [0.20, 0.40]

    all_base_results = []
    all_improved_results = []

    print("=" * 80)
    print("各プロジェクトのコストベネフィット分析")
    print("=" * 80)

    for project in projects:
        base_results, improved_results = analyze_project(
            project_name=project['name'],
            base_model_path=project['base_model'],
            improved_model_path=project['improved_model'],
            commit_metrics_path=project['commit_metrics'],
            effort_levels=effort_levels
        )
        all_base_results.append(base_results)
        all_improved_results.append(improved_results)

    df_test_results = perform_wilcoxon_test(
        all_base_results, all_improved_results, effort_levels, alpha=0.05
    )


if __name__ == "__main__":
    main()
