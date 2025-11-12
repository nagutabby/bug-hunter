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

def plot_cost_benefit_curve(df_metrics, save_path='cost_benefit_curve.png'):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(df_metrics['review_effort_ratio'] * 100,
            df_metrics['bug_detection_ratio'] * 100,
            'b-', linewidth=2, label='Improved')
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('Review Effort (%)', fontsize=12)
    ax.set_ylabel('Bug Detection Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"グラフを '{save_path}' に保存しました")
    plt.close()

def find_key_points(df_metrics):
    key_points = []

    for target_effort in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        idx = df_metrics[df_metrics['review_effort_ratio'] <= target_effort].index.max()
        if pd.notna(idx):
            row = df_metrics.loc[idx]
            key_points.append({
                'target': f'{int(target_effort*100)}%労力',
                'commits_reviewed': row['commits_reviewed'],
                'bugs_found': row['bugs_found'],
                'review_effort_ratio': row['review_effort_ratio'],
                'bug_detection_ratio': row['bug_detection_ratio'],
                'precision': row['precision']
            })

    return pd.DataFrame(key_points)

def main():
    project_name = "elasticsearch"
    model_path = f"../data/remove/{project_name}/predictions_add_method_commit_level_metrics.pkl"
    output_dir = f"../materials/images/{project_name}"

    print("=== モデルの読み込み ===")
    model_data = load_model(model_path)

    predictions_data = model_data['predictions_data']
    y_true = predictions_data['y_true']
    y_pred_proba = predictions_data['y_pred_proba']

    print(f"テストデータ数: {len(y_true)}")
    print(f"バグ総数: {np.sum(y_true)}")
    print(f"バグ比率: {np.mean(y_true):.2%}")

    print("\n=== Cost-Benefit Curve計算（予測確率降順） ===")
    df_metrics = calculate_cost_benefit_by_ranking(y_true, y_pred_proba)

    print("\n=== 主要ポイント ===")
    df_key_points = find_key_points(df_metrics)
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

    plot_cost_benefit_curve(df_metrics, save_path=f'{output_dir}/cost_benefit_curve.png')

if __name__ == "__main__":
    main()
