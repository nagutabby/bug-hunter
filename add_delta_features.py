import pandas as pd
from pydriller import Repository
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定項目 ---
INPUT_CSV_FILE = 'method-p_with_commit_date.csv'
OUTPUT_CSV_FILE = 'method-p_with_delta_features.csv'
MAX_WORKERS = 9  # 並列処理のワーカー数（CPUコア数に応じて調整）

# BugHunterデータセットの正確な15プロジェクトのリポジトリURLリスト
repo_urls = {
    "Android-Universal-Image-Loader": "https://github.com/nostra13/Android-Universal-Image-Loader.git",
    "antlr4": "https://github.com/antlr/antlr4.git",
    "BroadleafCommerce": "https://github.com/BroadleafCommerce/BroadleafCommerce.git",
    "ceylon-ide-eclipse": "https://github.com/ceylon/ceylon-ide-eclipse.git",
    "elasticsearch": "https://github.com/elastic/elasticsearch.git",
    "hazelcast": "https://github.com/hazelcast/hazelcast.git",
    "junit": "https://github.com/junit-team/junit4.git",
    "MapDB": "https://github.com/jankotek/MapDB.git",
    "mcMMO": "https://github.com/nossr50/mcMMO-1.git",
    "mct": "https://github.com/nasa/openmct.git",
    "neo4j": "https://github.com/neo4j/neo4j.git",
    "netty": "https://github.com/netty/netty.git",
    "orientdb": "https://github.com/orientechnologies/orientdb.git",
    "oryx": "https://github.com/OryxProject/oryx.git",
    "titan": "https://github.com/thinkaurelius/titan.git",
}

# スレッドセーフなカウンター用のロック
progress_lock = Lock()
total_processed = 0

def process_project_commits(project_name, repo_url, commit_hashes):
    """
    単一プロジェクトのコミット情報を処理する関数

    Args:
        project_name (str): プロジェクト名
        repo_url (str): リポジトリURL
        commit_hashes (list): 処理対象のコミットハッシュリスト

    Returns:
        dict: コミットハッシュをキーとするコミット情報の辞書
    """
    global total_processed

    project_commit_info = {}
    total_commits_to_find = len(commit_hashes)
    processed_count = 0

    logger.info(f"プロジェクト '{project_name}' の処理を開始 (対象のユニークなコミット数: {total_commits_to_find})")

    try:
        for commit in Repository(repo_url, only_commits=commit_hashes).traverse_commits():
            project_commit_info[commit.hash] = {
                'Insertions': commit.insertions,
                'Deletions': commit.deletions,
                'LinesChanged': commit.insertions + commit.deletions,
                'FilesChanged': commit.files,
                'LinesChangedPerFile': (commit.insertions + commit.deletions) / commit.files if commit.files > 0 else 0,
            }
            processed_count += 1

            # スレッドセーフなプログレス更新
            with progress_lock:
                total_processed += 1
                if processed_count % 10 == 0 or processed_count == total_commits_to_find:
                    logger.info(f"プロジェクト '{project_name}': {processed_count}/{total_commits_to_find} 件完了")

        logger.info(f"プロジェクト '{project_name}' の情報取得が完了しました。")
        return project_name, project_commit_info

    except Exception as e:
        logger.error(f"プロジェクト '{project_name}' の処理中にエラーが発生: {e}")
        return project_name, {}

def process_commits_batch(commit_hashes_batch, repo_url):
    """
    コミットハッシュのバッチを処理する関数（より細かな並列化用）

    Args:
        commit_hashes_batch (list): 処理するコミットハッシュのバッチ
        repo_url (str): リポジトリURL

    Returns:
        dict: コミット情報の辞書
    """
    batch_commit_info = {}

    try:
        for commit in Repository(repo_url, only_commits=commit_hashes_batch).traverse_commits():
            batch_commit_info[commit.hash] = {
                'Insertions': commit.insertions,
                'Deletions': commit.deletions,
                'LinesChanged': commit.insertions + commit.deletions,
                'FilesChanged': commit.files,
                'LinesChangedPerFile': (commit.insertions + commit.deletions) / commit.files,
            }

        return batch_commit_info

    except Exception as e:
        logger.error(f"バッチ処理中にエラーが発生: {e}")
        return {}

def split_into_batches(items, batch_size):
    """
    リストを指定サイズのバッチに分割する

    Args:
        items (list): 分割対象のリスト
        batch_size (int): バッチサイズ

    Returns:
        list: バッチのリスト
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def main():
    """
    並列処理を使用してCSVファイルからコミット情報を取得し、新しいCSVファイルとして保存するメイン関数
    """
    global total_processed

    script_start_time = time.time()
    logger.info("並列処理スクリプトを開始します...")

    if not os.path.exists(INPUT_CSV_FILE):
        logger.error(f"入力ファイル '{INPUT_CSV_FILE}' が見つかりません。")
        sys.exit(1)

    logger.info(f"入力ファイル '{INPUT_CSV_FILE}' を読み込んでいます...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        logger.info(f"読み込み完了。{len(df)}行のデータを検出しました。")
    except Exception as e:
        logger.error(f"CSVファイルの読み込みに失敗しました: {e}")
        sys.exit(1)

    commit_info_cache = {}
    projects_in_csv = df['Project'].unique()

    logger.info("CSVファイルに含まれるプロジェクト:")
    for i, project in enumerate(projects_in_csv, 1):
        status = "✓" if project in repo_urls else "✗ (URLなし)"
        logger.info(f"  {i:2d}. {project} {status}")

    # 処理対象のプロジェクトとコミットハッシュを準備
    project_tasks = []
    for project_name in projects_in_csv:
        if project_name in repo_urls:
            repo_url = repo_urls[project_name]
            commit_hashes = df[df['Project'] == project_name]['Hash'].dropna().unique().tolist()
            if commit_hashes:
                project_tasks.append((project_name, repo_url, commit_hashes))

    logger.info(f"\n並列処理を開始します (ワーカー数: {MAX_WORKERS})")
    logger.info(f"処理対象プロジェクト数: {len(project_tasks)}")

    # 並列処理実行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # プロジェクト単位で並列処理を実行
        future_to_project = {
            executor.submit(process_project_commits, project_name, repo_url, commit_hashes): project_name
            for project_name, repo_url, commit_hashes in project_tasks
        }

        # 結果を収集
        for future in as_completed(future_to_project):
            project_name = future_to_project[future]
            try:
                returned_project_name, project_commit_info = future.result()
                commit_info_cache.update(project_commit_info)
                logger.info(f"プロジェクト '{returned_project_name}' の処理が完了 (取得件数: {len(project_commit_info)})")
            except Exception as e:
                logger.error(f"プロジェクト '{project_name}' の処理でエラーが発生: {e}")

    logger.info("全プロジェクトのコミット情報取得が完了しました。")
    logger.info(f"取得したコミット情報: {len(commit_info_cache)} 件")
    logger.info("取得したデータを元のDataFrameにマージしています...")

    # DataFrameにマージ
    commit_df = pd.DataFrame.from_dict(commit_info_cache, orient='index')
    df_merged = pd.merge(df, commit_df, left_on='Hash', right_index=True, how='left')
    logger.info("マージ完了。")

    # 結果を保存
    logger.info(f"最終結果を '{OUTPUT_CSV_FILE}' に書き込んでいます...")
    try:
        df_merged.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        logger.info("書き込み完了。")
    except Exception as e:
        logger.error(f"ファイル '{OUTPUT_CSV_FILE}' の保存中にエラーが発生: {e}")
        sys.exit(1)

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    logger.info("=" * 60)
    logger.info("全ての処理が正常に完了しました！")
    logger.info(f"出力ファイル: {OUTPUT_CSV_FILE}")
    logger.info(f"合計処理時間: {total_time:.2f} 秒")
    logger.info(f"コミット情報取得数: {len(commit_info_cache)} 件")
    logger.info(f"並列処理ワーカー数: {MAX_WORKERS}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
