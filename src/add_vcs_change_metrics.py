import pandas as pd
import os
import sys
from git import Repo

def get_commit_change_stats(repo_path, commit_hash):
    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_hash)

        if not commit.parents:
            print(f"  警告: コミット {commit_hash} には親コミットがありません（初期コミット）")
            return None

        parent = commit.parents[0]

        files_changed = 0
        lines_added = 0
        lines_deleted = 0
        total_lines_before = 0

        diff_index = parent.diff(commit)

        for diff_item in diff_index:
            if diff_item.change_type in ['A', 'M', 'D', 'R']:
                files_changed += 1

            if diff_item.change_type == 'M':
                try:
                    a_content = diff_item.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                    b_content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore')

                    a_lines = a_content.splitlines()
                    b_lines = b_content.splitlines()

                    total_lines_before += len(a_lines)

                    import difflib
                    diff = difflib.unified_diff(a_lines, b_lines, lineterm='')

                    for line in diff:
                        if line.startswith('+') and not line.startswith('+++'):
                            lines_added += 1
                        elif line.startswith('-') and not line.startswith('---'):
                            lines_deleted += 1
                except Exception as e:
                    print(f"  警告: ファイル比較中にエラー: {e}")
                    pass

            elif diff_item.change_type == 'A':
                try:
                    b_content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                    b_lines = b_content.splitlines()
                    lines_added += len(b_lines)
                except:
                    pass

            elif diff_item.change_type == 'D':
                try:
                    a_content = diff_item.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                    a_lines = a_content.splitlines()
                    total_lines_before += len(a_lines)
                    lines_deleted += len(a_lines)
                except:
                    pass

        return {
            'files_changed': files_changed,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'total_lines_before': total_lines_before
        }

    except Exception as e:
        print(f"  エラー: コミット {commit_hash} の統計取得中に例外が発生しました: {e}")
        return None

def calculate_vcs_metrics(stats):
    if stats is None:
        return {
            'num_files': None,
            'lines_added_ratio': None,
            'lines_deleted_ratio': None,
            'lines_per_file': None
        }

    nf = stats['files_changed']
    la = stats['lines_added']
    ld = stats['lines_deleted']
    lt = stats['total_lines_before']

    la_lt = la / lt if lt > 0 else None
    ld_lt = ld / lt if lt > 0 else None
    lt_nf = lt / nf if nf > 0 else None

    return {
        'num_files': nf,
        'lines_added_ratio': la_lt,
        'lines_deleted_ratio': ld_lt,
        'lines_per_file': lt_nf
    }

def prepare_enhanced_csv_output(original_df, metrics_results):
    enhanced_df = original_df.copy()

    new_columns = [
        'num_files',
        'lines_added_ratio',
        'lines_deleted_ratio',
        'lines_per_file'
    ]

    for col in new_columns:
        enhanced_df[col] = pd.NA

    for record_id, metrics in metrics_results.items():
        row_index = record_id - 1

        if metrics is not None:
            enhanced_df.loc[row_index, 'num_files'] = metrics['num_files']
            enhanced_df.loc[row_index, 'lines_added_ratio'] = metrics['lines_added_ratio']
            enhanced_df.loc[row_index, 'lines_deleted_ratio'] = metrics['lines_deleted_ratio']
            enhanced_df.loc[row_index, 'lines_per_file'] = metrics['lines_per_file']

    return enhanced_df

def main():
    base_dir = "../data/remove/elasticsearch/"
    input_path = base_dir + "method-p_add_change_metrics.csv"
    output_path = base_dir + "method-p_add_vcs_change_metrics.csv"
    repo_path = "/Users/nagutabby/elasticsearch"
    max_records = 3000

    if not os.path.exists(input_path):
        print(f"エラー: CSVファイル '{input_path}' が見つかりません")
        sys.exit(1)

    if not os.path.exists(repo_path):
        print(f"エラー: リポジトリパス '{repo_path}' が見つかりません")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)

        required_columns = ['Hash']
        for col in required_columns:
            if col not in df.columns:
                print(f"エラー: CSVファイルに{col}カラムが見つかりません")
                sys.exit(1)

        if len(df) == 0:
            print("エラー: CSVファイルにデータが含まれていません")
            sys.exit(1)

        if len(df) > max_records:
            print(f"情報: CSVファイルには{len(df)}レコードありますが、最初の{max_records}レコードのみ処理します")
            df_to_process = df.head(max_records)
        else:
            print(f"情報: CSVファイルの全{len(df)}レコードを処理します")
            df_to_process = df

        metrics_results = {}
        processed_count = 0

        print(f"\n{'='*80}")
        print(f"処理開始: {len(df_to_process)}レコードを処理します")
        print(f"{'='*80}")

        for idx, record in df_to_process.iterrows():
            commit_hash = record['Hash']
            record_id = idx + 1

            print(f"\n処理中のレコード {record_id}/{len(df_to_process)}: Hash={commit_hash}")

            print(f"  コミット統計を取得中...")
            stats = get_commit_change_stats(repo_path, commit_hash)
            metrics = calculate_vcs_metrics(stats)

            if stats:
                print(f"  [DEBUG] LA (追加行数): {stats['lines_added']}")
                print(f"  [DEBUG] LD (削除行数): {stats['lines_deleted']}")
                print(f"  [DEBUG] LT (変更前総行数): {stats['total_lines_before']}")
                print(f"  NF (変更ファイル数): {metrics['num_files']}")
                print(f"  LA/LT (追加行数比): {metrics['lines_added_ratio']:.4f}" if metrics['lines_added_ratio'] is not None else "  LA/LT: N/A")
                print(f"  LD/LT (削除行数比): {metrics['lines_deleted_ratio']:.4f}" if metrics['lines_deleted_ratio'] is not None else "  LD/LT: N/A")
                print(f"  LT/NF (ファイル毎行数): {metrics['lines_per_file']:.2f}" if metrics['lines_per_file'] is not None else "  LT/NF: N/A")

            metrics_results[record_id] = metrics
            processed_count += 1

        enhanced_df = prepare_enhanced_csv_output(df, metrics_results)
        enhanced_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\n{'='*80}")
        print(f"VCS変更メトリクスが '{output_path}' に保存されました")
        print(f"\n=== 処理完了サマリー ===")
        print(f"対象レコード数: {len(df_to_process)}")
        print(f"処理されたレコード数: {processed_count}")
        print(f"成功率: {processed_count/len(df_to_process)*100:.1f}%")
        print(f"\n出力ファイル: {output_path}")

    except Exception as e:
        print(f"エラー: メイン処理中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
