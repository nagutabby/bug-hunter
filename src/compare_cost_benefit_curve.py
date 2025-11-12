import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import JavaCodeTokenizer

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def calculate_cost_benefit_by_ranking(y_true, y_pred_proba):
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]

    total_commits = len(y_true)
    total_bugs = np.sum(y_true)

    results = []
    bugs_found = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            bugs_found += 1

        commits_reviewed = i + 1
        review_effort_ratio = commits_reviewed / total_commits
        bug_detection_ratio = bugs_found / total_bugs if total_bugs > 0 else 0

        results.append({
            'commits_reviewed': commits_reviewed,
            'bugs_found': bugs_found,
            'review_effort_ratio': review_effort_ratio,
            'bug_detection_ratio': bug_detection_ratio,
            'precision': bugs_found / commits_reviewed if commits_reviewed > 0 else 0
        })

    return pd.DataFrame(results)

def find_key_points(df_metrics, model_name):
    key_points = []

    for target_effort in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        idx = df_metrics[df_metrics['review_effort_ratio'] <= target_effort].index.max()
        if pd.notna(idx):
            row = df_metrics.loc[idx]
            key_points.append({
                'model': model_name,
                'target': f'{int(target_effort*100)}%労力',
                'commits_reviewed': row['commits_reviewed'],
                'bugs_found': row['bugs_found'],
                'review_effort_ratio': row['review_effort_ratio'],
                'bug_detection_ratio': row['bug_detection_ratio'],
                'precision': row['precision']
            })

    return pd.DataFrame(key_points)

def plot_comparison_cost_benefit_curve(df_base, df_improved, save_path='comparison_cost_benefit_curve.png'):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(df_base['review_effort_ratio'] * 100,
            df_base['bug_detection_ratio'] * 100,
            'r-', linewidth=2, label='Base', alpha=0.7)
    ax.plot(df_improved['review_effort_ratio'] * 100,
            df_improved['bug_detection_ratio'] * 100,
            'b-', linewidth=2, label='Improved', alpha=0.7)
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('Review Effort (%)', fontsize=12)
    ax.set_ylabel('Bug Detection Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
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

def main():
    project_name = "elasticsearch"
    base_model_path = f"../data/remove/{project_name}/predictions_base.pkl"
    improved_model_path = f"../data/remove/{project_name}/predictions_add_method_commit_level_metrics.pkl"
    output_dir = f"../materials/images/{project_name}"

    print("=== モデルの読み込み ===")
    print(f"Base Model: {base_model_path}")
    print(f"  (訓練データ: method-p_drop_columns_rows.csv)")
    model_base = load_model(base_model_path)
    print(f"Improved Model: {improved_model_path}")
    print(f"  (訓練データ: method-p_add_method_commit_level_metrics.csv)")
    model_improved = load_model(improved_model_path)

    y_true_base = model_base['predictions_data']['y_true']
    y_pred_proba_base = model_base['predictions_data']['y_pred_proba']

    y_true_improved = model_improved['predictions_data']['y_true']
    y_pred_proba_improved = model_improved['predictions_data']['y_pred_proba']

    print(f"\nBase Model - テストデータ数: {len(y_true_base)}, バグ数: {np.sum(y_true_base)}")
    print(f"Improved Model - テストデータ数: {len(y_true_improved)}, バグ数: {np.sum(y_true_improved)}")

    print("\n=== Cost-Benefit Curve計算 ===")
    df_base = calculate_cost_benefit_by_ranking(y_true_base, y_pred_proba_base)
    df_improved = calculate_cost_benefit_by_ranking(y_true_improved, y_pred_proba_improved)

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

    plot_comparison_cost_benefit_curve(df_base, df_improved,
                                       save_path=f'{output_dir}/comparison_cost_benefit_curve.png')

    print("\n" + "="*60)
    print("まとめ")
    print("="*60)
    print("Base Model:")
    print("  訓練データ: method-p_drop_columns_rows.csv")
    print("Improved Model:")
    print("  訓練データ: method-p_add_method_commit_level_metrics.csv")
    print("  改善手法: メソッドレベルとコミットレベルのメトリクスを追加")
    print("="*60)

if __name__ == "__main__":
    main()
