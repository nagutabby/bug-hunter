import pandas as pd
import re
import os
from dotenv import load_dotenv
from pydriller import Repository
from github import Github

class CommitPRAnalyzer:
    def __init__(self, local_repo_path, github_repo_name, github_token):
        self.local_repo_path = local_repo_path
        self.github_repo_name = github_repo_name
        self.github_token = github_token
        self.g = Github(github_token)
        self.github_repo = self.g.get_repo(github_repo_name)
        self.api_call_count = 0

    def should_check_rate_limit(self):
        return self.api_call_count % 100 == 0

    def check_rate_limit(self):
        try:
            rate_limit = self.g.get_rate_limit()
            core_limit = rate_limit.resources.core

            print(f"\n--- GitHub APIレートリミット状況 ---")
            print(f"残り回数: {core_limit.remaining}/{core_limit.limit}")
            print(f"リセット時刻: {core_limit.reset}")

            if core_limit.remaining < 100:
                print(f"⚠️  警告: 残り回数が少なくなっています ({core_limit.remaining}回)")

            if core_limit.remaining == 0:
                print(f"❌ レートリミットに達しました。{core_limit.reset}まで待機が必要です")
                return False

            return True

        except Exception as e:
            print(f"レートリミット確認エラー: {e}")
            print("レートリミット確認をスキップして処理を続行します")
            return True

    def extract_pr_number_from_commit_message(self, commit_message):
        patterns = [
            r'#(\d+)',
            r'pull/(\d+)',
            r'PR\s*#?(\d+)',
            r'Merge pull request #(\d+)',
            r'\(#(\d+)\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, commit_message, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def get_commit_message(self, commit_hash):
        try:
            for commit in Repository(self.local_repo_path, single=commit_hash).traverse_commits():
                return commit.msg
        except Exception as e:
            print(f"コミット取得エラー: {e}")
            return None

    def get_pr_info(self, pr_number):
        try:
            self.api_call_count += 1

            if self.should_check_rate_limit():
                if not self.check_rate_limit():
                    print("レートリミットのため処理を停止します")
                    return None

            pr = self.github_repo.get_pull(pr_number)

            return {
                'title': pr.title,
                'number': pr.number,
                'author': pr.user.login,
                'state': pr.state,
                'created_at': pr.created_at,
                'merged_at': pr.merged_at,
                'url': pr.html_url,
                'body': pr.body
            }

        except Exception as e:
            print(f"PR情報取得エラー: {e}")
            return None

    def display_commit_and_pr_info(self, commit_hash, commit_message, pr_info):
        print(f"\n{'='*60}")
        print(f"コミットハッシュ: {commit_hash}")
        print(f"コミットメッセージ: {commit_message}")

        if pr_info:
            print(f"\n--- PR情報 ---")
            print(f"タイトル: {pr_info['title']}")
            print(f"PR番号: #{pr_info['number']}")
            print(f"作成者: {pr_info['author']}")
            print(f"状態: {pr_info['state']}")
            print(f"作成日: {pr_info['created_at']}")
            print(f"マージ日: {pr_info['merged_at']}")
            print(f"URL: {pr_info['url']}")
            print(f"\n--- PR説明 ---")
            print(pr_info['body'] if pr_info['body'] else "説明なし")
        else:
            print("PR情報が見つかりませんでした")

        print(f"\nAPI呼び出し回数: {self.api_call_count}")

    def analyze_csv(self, csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            print(f"CSVファイル読み込み完了: {len(df)}行")

            if 'Hash' not in df.columns or 'Number of Bugs' not in df.columns:
                print("エラー: 'Hash'または'Number of Bugs'カラムが見つかりません")
                return

            zero_bug_commits = df[df['Number of Bugs'] == 0]
            print(f"Number of Bugs = 0のコミット: {len(zero_bug_commits)}件")

            if len(zero_bug_commits) == 0:
                print("条件に合致するコミットが見つかりませんでした")
                return

            processed_count = 0

            for index, row in zero_bug_commits.iterrows():
                commit_hash = row['Hash']

                print(f"\n処理中のコミット {processed_count + 1}/{len(zero_bug_commits)}: {commit_hash}")

                commit_message = self.get_commit_message(commit_hash)
                if not commit_message:
                    print("コミットメッセージの取得に失敗しました")
                    continue

                pr_number = self.extract_pr_number_from_commit_message(commit_message)
                pr_info = None

                if pr_number:
                    pr_info = self.get_pr_info(pr_number)

                self.display_commit_and_pr_info(commit_hash, commit_message, pr_info)

                processed_count += 1

                if processed_count < len(zero_bug_commits):
                    while True:
                        continue_choice = input(f"\n次のコミットを処理しますか？ (y/n): ").strip().lower()
                        if continue_choice in ['y', 'yes', 'n', 'no']:
                            break
                        print("'y'または'n'で入力してください")

                    if continue_choice in ['n', 'no']:
                        print("処理を終了します")
                        break
                else:
                    print("\nすべてのコミットの処理が完了しました")

        except Exception as e:
            print(f"CSV処理エラー: {e}")

def main():
    load_dotenv()

    LOCAL_REPO_PATH = "/Users/nagutabby/elasticsearch"
    GITHUB_REPO_NAME = "elastic/elasticsearch"
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    CSV_FILE_PATH = "../data/remove/elasticsearch/method-p.csv"

    if not GITHUB_TOKEN:
        print("エラー: .envファイルでGITHUB_TOKENが設定されていません")
        return

    analyzer = CommitPRAnalyzer(LOCAL_REPO_PATH, GITHUB_REPO_NAME, GITHUB_TOKEN)

    print("GitHub API接続確認中...")
    if analyzer.check_rate_limit():
        print("GitHub API接続成功")
        analyzer.analyze_csv(CSV_FILE_PATH)
    else:
        print("GitHub APIレートリミットの問題により処理を開始できません")

if __name__ == "__main__":
    main()
