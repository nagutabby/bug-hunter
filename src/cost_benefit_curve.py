import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def calculate_cost_benefit_greedy(y_true, y_pred_proba, efforts, capacity_ratios):
    total_effort = np.sum(efforts)
    total_bugs = np.sum(y_true)

    results = []

    results.append({
        'capacity_ratio': 0.0,
        'effort_used': 0.0,
        'commits_reviewed': 0,
        'bugs_found': 0.0,
        'review_effort_ratio': 0.0,
        'bug_detection_ratio': 0.0
    })

    for capacity_ratio in capacity_ratios:
        capacity = total_effort * capacity_ratio

        selected_indices, total_value, total_weight = knapsack_greedy(
            weights=efforts,
            values=y_pred_proba,
            capacity=capacity
        )

        bugs_found = np.sum(y_true[selected_indices])

        results.append({
            'capacity_ratio': capacity_ratio,
            'effort_used': total_weight,
            'commits_reviewed': len(selected_indices),
            'bugs_found': bugs_found,
            'review_effort_ratio': total_weight / total_effort if total_effort > 0 else 0,
            'bug_detection_ratio': bugs_found / total_bugs if total_bugs > 0 else 0
        })

    return pd.DataFrame(results)

def find_key_points(df_metrics, model_name):
    key_points = []

    for _, row in df_metrics.iterrows():
        ratio_pct = round(row['capacity_ratio'] * 100)

        if ratio_pct > 0 and ratio_pct % 20 == 0:
            key_points.append({
                'model': model_name,
                'target': f"{ratio_pct}%労力",
                'commits_reviewed': int(row['commits_reviewed']),
                'bugs_found': int(row['bugs_found']),
                'review_effort_ratio': row['review_effort_ratio'],
                'bug_detection_ratio': row['bug_detection_ratio'],
                'precision': row['bugs_found'] / row['commits_reviewed'] if row['commits_reviewed'] > 0 else 0
            })

    return pd.DataFrame(key_points)

def plot_single_curve(df_data, save_path='cost_benefit_curve.png'):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.plot(df_data['review_effort_ratio'] * 100,
            df_data['bug_detection_ratio'] * 100,
            'b-', linewidth=2.5, label='Improved')

    ax.set_xlabel('Review Effort (%)', fontsize=20)
    ax.set_ylabel('Bug Detection Rate (%)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"グラフを '{save_path}' に保存しました")
    plt.close()

def plot_comparison_curve(df_base, df_improved, save_path='comparison_cost_benefit_curve.png'):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(df_base['review_effort_ratio'] * 100,
            df_base['bug_detection_ratio'] * 100,
            'r-', linewidth=2, label='Base', alpha=0.7)
    ax.plot(df_improved['review_effort_ratio'] * 100,
            df_improved['bug_detection_ratio'] * 100,
            'b-', linewidth=2, label='Improved', alpha=0.7)
    ax.set_xlabel('Review Effort (%)', fontsize=20)
    ax.set_ylabel('Bug Detection Rate (%)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"比較グラフを '{save_path}' に保存しました")
    plt.close()

def calculate_improvement(df_base_key, df_improved_key):
    improvements = []

    for target in df_base_key['target'].unique():
        base_row = df_base_key[df_base_key['target'] == target]
        improved_row = df_improved_key[df_improved_key['target'] == target]

        if len(base_row) > 0 and len(improved_row) > 0:
            base_detection = base_row['bug_detection_ratio'].values[0]
            improved_detection = improved_row['bug_detection_ratio'].values[0]

            detection_improvement = (improved_detection - base_detection) * 100
            bugs_improvement = improved_row['bugs_found'].values[0] - base_row['bugs_found'].values[0]

            improvements.append({
                'target': target,
                'base_detection': base_detection * 100,
                'improved_detection': improved_detection * 100,
                'detection_improvement_pct': detection_improvement,
                'bugs_improvement': bugs_improvement
            })

    return pd.DataFrame(improvements)

def analyze_single_model(model_path, commit_metrics_path, output_dir):
    print("=== モデルの読み込み ===")
    model_data = load_model(model_path)

    predictions_data = model_data['predictions_data']
    y_true = predictions_data['y_true']
    y_pred_proba = predictions_data['y_pred_proba']

    print(f"テストデータ数: {len(y_true)}")
    print(f"バグ総数: {np.sum(y_true)}")
    print(f"バグ比率: {np.mean(y_true):.2%}")

    print("\n=== コミットレベルメトリクスの読み込み ===")
    df_commits = pd.read_csv(commit_metrics_path)

    test_indices = predictions_data.get('test_indices', None)
    if test_indices is not None:
        df_test = df_commits.iloc[test_indices].reset_index(drop=True)
    else:
        print("警告: test_indicesが見つかりません。全データを使用します")
        df_test = df_commits

    print(f"テストセットのコミット数: {len(df_test)}")

    print("\n=== 労力の計算 ===")
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

    sorted_efforts = np.sort(efforts)
    lower_80_percent_count = int(len(sorted_efforts) * 0.8)
    total_effort_lower_80 = np.sum(sorted_efforts[:lower_80_percent_count])

    print(f"全コミットの総労力: {np.sum(efforts):.2f}")
    print(f"下位80%のコミット数: {lower_80_percent_count}")
    print(f"下位80%の総労力（レビュー可能な総労力）: {total_effort_lower_80:.2f}")
    print(f"平均労力: {np.mean(efforts):.2f}")

    print("\n=== 貪欲法による近似解の計算 ===")

    capacity_ratios_for_calculation = [i * 0.01 for i in range(1, 101)]

    results = []
    total_bugs = np.sum(y_true)

    results.append({
        'capacity_ratio': 0.0,
        'effort_used': 0.0,
        'commits_reviewed': 0,
        'bugs_found': 0.0,
        'review_effort_ratio': 0.0,
        'bug_detection_ratio': 0.0
    })

    for capacity_ratio in capacity_ratios_for_calculation:
        capacity = total_effort_lower_80 * capacity_ratio

        selected_indices, total_value, total_weight = knapsack_greedy(
            weights=efforts,
            values=y_pred_proba,
            capacity=capacity
        )

        bugs_found = np.sum(y_true[selected_indices])

        results.append({
            'capacity_ratio': capacity_ratio,
            'effort_used': total_weight,
            'commits_reviewed': len(selected_indices),
            'bugs_found': bugs_found,
            'review_effort_ratio': total_weight / total_effort_lower_80 if total_effort_lower_80 > 0 else 0,
            'bug_detection_ratio': bugs_found / total_bugs if total_bugs > 0 else 0
        })

    df_results = pd.DataFrame(results)

    print("\n=== 主要ポイント ===")
    df_key_points = find_key_points(df_results, 'Model')
    print(df_key_points.to_string(index=False))

    print("\n=== 具体的な労力削減効果 ===")
    for _, row in df_key_points.iterrows():
        review_reduction = (1 - row['review_effort_ratio']) * 100
        print(f"\n【{row['target']}】")
        print(f"  レビュー対象: {row['commits_reviewed']}件")
        print(f"  バグ検出数: {row['bugs_found']}件")
        print(f"  レビュー労力: {row['review_effort_ratio']*100:.1f}% (削減: {review_reduction:.1f}%)")
        print(f"  バグ検出率: {row['bug_detection_ratio']*100:.1f}%")
        print(f"  Precision: {row['precision']:.3f}")

    print("\n=== グラフの生成 ===")
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_single_curve(
        df_data=df_results,
        save_path=f'{output_dir}/cost_benefit_curve.png'
    )

def compare_models(base_model_path, improved_model_path, commit_metrics_path, output_dir):
    print("=== モデルの読み込み ===")
    print(f"Base Model: {base_model_path}")
    model_base = load_model(base_model_path)
    print(f"Improved Model: {improved_model_path}")
    model_improved = load_model(improved_model_path)

    y_true_base = model_base['predictions_data']['y_true']
    y_pred_proba_base = model_base['predictions_data']['y_pred_proba']

    y_true_improved = model_improved['predictions_data']['y_true']
    y_pred_proba_improved = model_improved['predictions_data']['y_pred_proba']

    print(f"\nBase Model - テストデータ数: {len(y_true_base)}, バグ数: {np.sum(y_true_base)}")
    print(f"Improved Model - テストデータ数: {len(y_true_improved)}, バグ数: {np.sum(y_true_improved)}")

    print("\n=== コミットレベルメトリクスの読み込み ===")
    df_commits = pd.read_csv(commit_metrics_path)

    test_indices = model_improved['predictions_data'].get('test_indices', None)
    if test_indices is not None:
        df_test = df_commits.iloc[test_indices].reset_index(drop=True)
    else:
        print("警告: test_indicesが見つかりません。全データを使用します")
        df_test = df_commits

    print(f"テストセットのコミット数: {len(df_test)}")

    print("\n=== 労力の計算 ===")
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

    sorted_efforts = np.sort(efforts)
    lower_80_percent_count = int(len(sorted_efforts) * 0.8)
    total_effort_lower_80 = np.sum(sorted_efforts[:lower_80_percent_count])

    print(f"全コミットの総労力: {np.sum(efforts):.2f}")
    print(f"下位80%のコミット数: {lower_80_percent_count}")
    print(f"下位80%の総労力（レビュー可能な総労力）: {total_effort_lower_80:.2f}")
    print(f"平均労力: {np.mean(efforts):.2f}")

    print("\n=== Cost-Benefit Curve計算 ===")

    capacity_ratios_for_calculation = [i * 0.01 for i in range(1, 101)]

    def calculate_for_model(y_true, y_pred_proba):
        results = []
        total_bugs = np.sum(y_true)

        results.append({
            'capacity_ratio': 0.0,
            'effort_used': 0.0,
            'commits_reviewed': 0,
            'bugs_found': 0.0,
            'review_effort_ratio': 0.0,
            'bug_detection_ratio': 0.0
        })

        for capacity_ratio in capacity_ratios_for_calculation:
            capacity = total_effort_lower_80 * capacity_ratio

            selected_indices, total_value, total_weight = knapsack_greedy(
                weights=efforts,
                values=y_pred_proba,
                capacity=capacity
            )

            bugs_found = np.sum(y_true[selected_indices])

            results.append({
                'capacity_ratio': capacity_ratio,
                'effort_used': total_weight,
                'commits_reviewed': len(selected_indices),
                'bugs_found': bugs_found,
                'review_effort_ratio': total_weight / total_effort_lower_80 if total_effort_lower_80 > 0 else 0,
                'bug_detection_ratio': bugs_found / total_bugs if total_bugs > 0 else 0
            })

        return pd.DataFrame(results)

    df_base = calculate_for_model(y_true_base, y_pred_proba_base)
    df_improved = calculate_for_model(y_true_improved, y_pred_proba_improved)

    print("\n=== 主要ポイント抽出 ===")
    df_base_key = find_key_points(df_base, 'Base')
    df_improved_key = find_key_points(df_improved, 'Improved')

    df_comparison = pd.concat([df_base_key, df_improved_key], ignore_index=True)
    print(df_comparison.to_string(index=False))

    print("\n=== 性能改善の定量評価 ===")
    df_improvements = calculate_improvement(df_base_key, df_improved_key)

    for _, row in df_improvements.iterrows():
        if pd.notna(row.get('detection_improvement_pct')):
            print(f"\n【{row['target']}】")
            print(f"  Base: {row['base_detection']:.1f}% 検出")
            print(f"  Improved: {row['improved_detection']:.1f}% 検出")
            print(f"  改善: +{row['detection_improvement_pct']:.1f}%ポイント (+{int(row['bugs_improvement'])}件)")

    print("\n=== グラフの生成 ===")
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_comparison_curve(df_base, df_improved,
                         save_path=f'{output_dir}/comparison_cost_benefit_curve.png')

def main():
    project_name = "neo4j"

    base_model_path = f"../data/remove/{project_name}/predictions_base.pkl"
    improved_model_path = f"../data/remove/{project_name}/predictions_add_method_commit_level_metrics.pkl"
    commit_metrics_path = f"../data/remove/{project_name}/method-p_add_method_commit_level_metrics.csv"
    output_dir = f"../materials/images/{project_name}"

    print("=" * 80)
    print("単一モデル分析")
    print("=" * 80)
    analyze_single_model(
        model_path=improved_model_path,
        commit_metrics_path=commit_metrics_path,
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("モデル比較分析")
    print("=" * 80)
    compare_models(
        base_model_path=base_model_path,
        improved_model_path=improved_model_path,
        commit_metrics_path=commit_metrics_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
