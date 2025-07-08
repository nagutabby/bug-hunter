#!/usr/bin/env python3
import pandas as pd
import os
import sys
import re
import glob
from pathlib import Path
from lizard import analyze_file

def extract_class_name(parent_path):
    """
    Parentカラムの値からクラス名を抽出する（匿名内部クラス対応版）
    例: org.elasticsearch.ExceptionSerializationTests$1UnknownException
    → ExceptionSerializationTests
    """
    if not parent_path or pd.isna(parent_path):
        return None

    # 最後のドット以降を取得
    class_part = parent_path.split('.')[-1]

    # $マークがある場合の処理
    if '$' in class_part:
        # $の前の部分を取得（外部クラス名）
        outer_class = class_part.split('$')[0]
        return outer_class

    return class_part

def extract_method_name(long_name):
    """
    LongNameカラムからメソッド名を抽出する（改良版 - より正確な内部クラス処理）
    """
    if not long_name or pd.isna(long_name):
        return None, None

    # メソッドシグネチャの括弧の位置を見つける
    paren_index = long_name.find('(')
    if paren_index == -1:
        return None, None

    # 括弧の前の部分を取得
    method_part = long_name[:paren_index]

    # 最後のドットの位置を見つける
    last_dot_index = method_part.rfind('.')
    if last_dot_index == -1:
        return None, None

    # メソッド名を抽出
    method_name = method_part[last_dot_index + 1:]

    # クラス名を抽出（$記号を含む場合の処理）
    class_part = method_part[:last_dot_index]
    class_name_part = class_part.split('.')[-1] if '.' in class_part else class_part

    # <init>はコンストラクタ
    if method_name == '<init>':
        if '$' in class_name_part:
            # 内部クラスの場合
            class_parts = class_name_part.split('$')

            # 最後の部分を取得（内部クラス名）
            inner_class_part = class_parts[-1]

            # 数字で始まる場合（匿名内部クラス）の処理
            if inner_class_part and inner_class_part[0].isdigit():
                # 数字以降の部分を取得
                match = re.match(r'^\d+(.+)$', inner_class_part)
                if match:
                    actual_class_name = match.group(1)
                else:
                    # 数字のみの場合は外部クラス名を使用
                    actual_class_name = class_parts[0] if len(class_parts) > 1 else inner_class_part
            else:
                actual_class_name = inner_class_part

            # 内部クラスのコンストラクタの場合、内部クラス名を返す
            return actual_class_name, 'constructor'
        else:
            # 通常のクラスのコンストラクタ
            return class_name_part, 'constructor'

    # <clinit>はスタティックイニシャライザ
    elif method_name == '<clinit>':
        return 'static_initializer', 'method'

    # 通常のメソッド
    else:
        return method_name, 'method'

def extract_package_path(parent_path):
    """
    Parentカラムからパッケージパスを抽出する
    例: org.elasticsearch.ExceptionSerializationTests$1UnknownException
    → org/elasticsearch/
    """
    if not parent_path or pd.isna(parent_path):
        return None

    # 最後のドット以前がパッケージパス
    parts = parent_path.split('.')
    if len(parts) <= 1:
        return None

    # 最後の部分（クラス名）を除いてパッケージパスを構築
    package_parts = parts[:-1]
    return '/'.join(package_parts) + '/'

def checkout_commit_with_pydriller(repo_path, commit_hash):
    """
    PyDrillerを使って指定されたコミットにチェックアウトする
    """
    try:
        from git import Repo
        repo = Repo(repo_path)

        # 現在のブランチ/コミットを保存
        original_head = repo.head.commit.hexsha

        # 指定されたコミットにチェックアウト
        repo.git.checkout(commit_hash)

        print(f"成功: コミット {commit_hash} にチェックアウトしました")
        return True, original_head

    except Exception as e:
        print(f"エラー: PyDrillerでのチェックアウト中に例外が発生しました: {e}")
        return False, None

def restore_original_head(repo_path, original_head):
    """
    元のコミット/ブランチに戻る
    """
    try:
        from git import Repo
        repo = Repo(repo_path)
        repo.git.checkout(original_head)
        print(f"元のコミット {original_head[:8]} に戻りました")
    except Exception as e:
        print(f"警告: 元のコミットに戻る際にエラーが発生しました: {e}")

def find_java_file_in_filesystem(repo_path, class_name, package_path):
    """
    ファイルシステムから対象のJavaファイルを検索する（チェックアウト後）
    """
    import glob

    try:
        search_patterns = []

        # パッケージパスがある場合の検索パターン
        if package_path:
            # パッケージパス + クラス名.java
            pattern1 = os.path.join(repo_path, "**", package_path, f"{class_name}.java")
            search_patterns.append(pattern1)

            # src/main/java/ + パッケージパス + クラス名.java（Maven構造）
            pattern2 = os.path.join(repo_path, "**", "src", "main", "java", package_path, f"{class_name}.java")
            search_patterns.append(pattern2)

            # src/test/java/ + パッケージパス + クラス名.java（テストファイル）
            pattern3 = os.path.join(repo_path, "**", "src", "test", "java", package_path, f"{class_name}.java")
            search_patterns.append(pattern3)

        # クラス名のみでの検索パターン
        pattern4 = os.path.join(repo_path, "**", f"{class_name}.java")
        search_patterns.append(pattern4)

        # 各パターンで検索
        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # 最初に見つかったファイルを返す
                return matches[0]

        return None

    except Exception as e:
        print(f"エラー: ファイルシステム検索中に例外が発生しました: {e}")
        return None

def analyze_java_file_with_lizard(file_path):
    """
    ファイルパスを指定してLizardで分析する
    """
    try:
        # Lizardで直接ファイルを分析
        analysis_result = analyze_file(file_path)

        methods = []
        for func in analysis_result.function_list:
            methods.append({
                'method_name': func.name,
                'ccn': func.cyclomatic_complexity,
                'length': func.length,
                'tokens': func.token_count,
                'params': func.parameter_count,
                'filename': file_path,
                'line_number': func.start_line
            })

        return methods

    except Exception as e:
        print(f"エラー: Lizard分析中に例外が発生しました: {e}")
        return []

def filter_methods_by_target(methods, target_method_names):
    """
    対象のメソッド名でフィルタリングする（改良版 - 過度なマッチングを防ぐ）
    """
    filtered_methods = []

    for method in methods:
        method_name = method['method_name']

        # 対象メソッドと一致するかチェック
        for target_method, method_type in target_method_names:
            matched = False

            if method_type == 'constructor':
                # コンストラクタの場合は、より厳密なマッチングを行う
                constructor_patterns = [
                    # 完全一致パターン
                    target_method,
                    f"{target_method}::{target_method}",
                ]

                # パターンマッチングを厳密に行う
                for pattern in constructor_patterns:
                    if method_name == pattern:
                        matched = True
                        break

                # 追加の厳密チェック：コンストラクタは通常クラス名と同じ名前
                if not matched:
                    # method_nameが "クラス名::クラス名" の形式の場合のみマッチ
                    if '::' in method_name:
                        parts = method_name.split('::')
                        if len(parts) == 2 and parts[0] == parts[1] and parts[0] == target_method:
                            matched = True
                    # または単純にクラス名と一致する場合（ただし他のメソッドと区別するため慎重に）
                    elif method_name == target_method:
                        matched = True

                # 匿名内部クラスの処理（数字で始まる場合）
                if not matched and method_name.startswith('(anonymous)::'):
                    anon_method = method_name.split('::')[-1]
                    if anon_method == target_method:
                        matched = True
                    elif anon_method and anon_method[0].isdigit():
                        clean_name = re.sub(r'^\d+', '', anon_method)
                        if clean_name == target_method:
                            matched = True

            else:
                # 通常のメソッドの場合は従来通り
                if (method_name == target_method or
                    method_name.endswith(f"::{target_method}") or
                    method_name.split('::')[-1] == target_method):
                    matched = True

                # 内部クラスのメソッドの場合の追加チェック
                if not matched and method_name.startswith('(anonymous)::'):
                    anon_method = method_name.split('::')[-1]
                    if anon_method == target_method:
                        matched = True

            if matched:
                filtered_methods.append({
                    **method,
                    'target_method': target_method,
                    'method_type': method_type,
                    'detected_method': method_name
                })
                break

    return filtered_methods

def search_method_in_file_content(java_file_path, target_method, method_type='method'):
    """
    ファイル内容を直接検索してメソッドの存在を確認（匿名内部クラス対応）
    """
    try:
        with open(java_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if method_type == 'constructor':
            # コンストラクタの場合は、クラス定義と組み合わせて検索
            constructor_patterns = [
                # 通常のコンストラクタパターン
                rf'class\s+{re.escape(target_method)}\s*.*\{{',
                rf'public\s+{re.escape(target_method)}\s*\(',
                rf'private\s+{re.escape(target_method)}\s*\(',
                rf'protected\s+{re.escape(target_method)}\s*\(',
                rf'{re.escape(target_method)}\s*\(',
                # 匿名内部クラスのパターン
                rf'new\s+\w+\s*\([^)]*\)\s*\{{[^}}]*class\s+{re.escape(target_method)}',
                rf'static\s+class\s+{re.escape(target_method)}',
                # Exception系の特別パターン
                rf'{re.escape(target_method)}\s+extends\s+\w+Exception',
                # 内部クラスのパターン
                rf'class\s+{re.escape(target_method)}\s+extends',
                rf'class\s+{re.escape(target_method)}\s+implements',
            ]
        else:
            # メソッドの定義パターン
            constructor_patterns = [
                rf'(public|private|protected|)\s+\w+\s+{re.escape(target_method)}\s*\(',
                rf'(public|private|protected|)\s+void\s+{re.escape(target_method)}\s*\(',
                rf'{re.escape(target_method)}\s*\(',
                rf'@Override\s+.*{re.escape(target_method)}\s*\(',
            ]

        for pattern in constructor_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if matches:
                return True

        return False

    except Exception as e:
        print(f"  ファイル内検索でエラー: {e}")
        return False

def analyze_with_improved_strategy(java_file_path, target_method_names, debug=False):
    """
    改良された戦略でメソッドを検索する
    """
    # 通常のLizard分析
    methods = analyze_java_file_with_lizard(java_file_path)

    if debug:
        print(f"    Lizardで検出された全メソッド ({len(methods)}個):")
        for i, method in enumerate(methods):
            print(f"      {i+1:2d}. '{method['method_name']}' (CCN: {method['ccn']})")

    # 改良されたフィルタリング
    filtered_methods = filter_methods_by_target(methods, target_method_names)

    if debug:
        print(f"    フィルタリング後のメソッド ({len(filtered_methods)}個):")
        for method in filtered_methods:
            print(f"      - '{method['detected_method']}' -> '{method['target_method']}'")

    if filtered_methods:
        return filtered_methods, "改良されたLizard分析"

    # フォールバック戦略：ファイル内容を直接検索してメソッドの存在を確認
    target_method = target_method_names[0][0]
    method_type = target_method_names[0][1]

    if search_method_in_file_content(java_file_path, target_method, method_type):
        if debug:
            print(f"    ファイル内検索でメソッド '{target_method}' を発見")
        return [{
            'method_name': f"(fallback)::{target_method}",
            'ccn': 1,
            'length': 1,
            'tokens': 1,
            'params': 0,
            'filename': java_file_path,
            'line_number': 1,
            'target_method': target_method,
            'method_type': method_type,
            'detected_method': f"(fallback)::{target_method}",
            'fallback_used': True
        }], "ファイル内検索（フォールバック）"

    return [], "すべての戦略で失敗"



def main():
    # 設定
    csv_file = "method-p_filtered_v2.csv"
    repo_path = "/Users/nagutabby/elasticsearch"
    output_csv = "method_metrics.csv"

    # 処理をスキップするかどうかのフラグ
    SKIP_MISSING_METHODS = False  # Trueにするとメソッドが見つからない場合はスキップ
    DEBUG_MODE = False  # Trueにするとデバッグ情報を表示

    # CSVファイルの存在確認
    if not os.path.exists(csv_file):
        print(f"エラー: CSVファイル '{csv_file}' が見つかりません")
        sys.exit(1)

    # リポジトリパスの存在確認
    if not os.path.exists(repo_path):
        print(f"エラー: リポジトリパス '{repo_path}' が見つかりません")
        sys.exit(1)

    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file)

        # 必要なカラムの存在確認
        required_columns = ['Hash', 'Parent', 'LongName']
        for col in required_columns:
            if col not in df.columns:
                print(f"エラー: CSVファイルに{col}カラムが見つかりません")
                sys.exit(1)

        if len(df) == 0:
            print("エラー: CSVファイルにデータが含まれていません")
            sys.exit(1)

        all_method_metrics = []
        original_heads = {}
        skipped_count = 0

        try:
            # 各レコードを処理
            for idx, record in df.iterrows():
                commit_hash = record['Hash']
                parent_path = record['Parent']
                long_name = record['LongName']

                print(f"\n処理中のレコード {idx + 1}/{len(df)}:")
                print(f"  Hash: {commit_hash}")
                print(f"  Parent: {parent_path}")
                print(f"  LongName: {long_name}")

                # クラス名を抽出
                class_name = extract_class_name(parent_path)
                if not class_name:
                    print("  警告: クラス名を抽出できませんでした")
                    if SKIP_MISSING_METHODS:
                        skipped_count += 1
                        continue
                    else:
                        raise ValueError("クラス名を抽出できませんでした")

                # メソッド名を抽出
                method_name, method_type = extract_method_name(long_name)
                if not method_name:
                    print("  警告: メソッド名を抽出できませんでした")
                    if SKIP_MISSING_METHODS:
                        skipped_count += 1
                        continue
                    else:
                        raise ValueError("メソッド名を抽出できませんでした")

                # パッケージパスを抽出
                package_path = extract_package_path(parent_path)

                print(f"  抽出されたクラス名: {class_name}")
                print(f"  抽出されたメソッド名: {method_name} ({method_type})")
                print(f"  パッケージパス: {package_path}")

                # 指定されたコミットにチェックアウト
                if repo_path not in original_heads:
                    success, original_head = checkout_commit_with_pydriller(repo_path, commit_hash)
                    if not success:
                        print("  エラー: コミットのチェックアウトに失敗しました")
                        if SKIP_MISSING_METHODS:
                            skipped_count += 1
                            continue
                        else:
                            raise ValueError("コミットのチェックアウトに失敗しました")
                    original_heads[repo_path] = original_head
                else:
                    try:
                        from git import Repo
                        repo = Repo(repo_path)
                        current_commit = repo.head.commit.hexsha
                        if current_commit != commit_hash:
                            success, _ = checkout_commit_with_pydriller(repo_path, commit_hash)
                            if not success:
                                print("  エラー: コミットのチェックアウトに失敗しました")
                                if SKIP_MISSING_METHODS:
                                    skipped_count += 1
                                    continue
                                else:
                                    raise ValueError("コミットのチェックアウトに失敗しました")
                    except Exception as e:
                        print(f"  エラー: 現在のコミット確認中にエラー: {e}")
                        if SKIP_MISSING_METHODS:
                            skipped_count += 1
                            continue
                        else:
                            raise

                # ファイルシステムからJavaファイルを検索
                java_file_path = find_java_file_in_filesystem(repo_path, class_name, package_path)

                if not java_file_path:
                    print(f"  警告: クラス '{class_name}' のJavaファイルが見つかりません")
                    if SKIP_MISSING_METHODS:
                        skipped_count += 1
                        continue
                    else:
                        raise ValueError(f"クラス '{class_name}' のJavaファイルが見つかりません")

                print(f"  見つかったJavaファイル: {java_file_path}")

                # Lizardで分析実行
                filtered_methods, strategy = analyze_with_improved_strategy(
                    java_file_path,
                    [(method_name, method_type)],
                    debug=DEBUG_MODE
                )
                print(f"  戦略: {strategy}")

                # メソッドが見つからなかった場合の処理（改良版）
                if len(filtered_methods) == 0:
                    print("  警告: 対象メソッドが見つかりませんでした")
                    print("  詳細デバッグ情報:")
                    print(f"    対象メソッド名: {method_name}")
                    print(f"    メソッドタイプ: {method_type}")
                    print(f"    ファイル: {java_file_path}")
                    print(f"    元のLongName: {long_name}")

                    # 通常のLizard分析結果を表示
                    methods = analyze_java_file_with_lizard(java_file_path)
                    print(f"    検出された全メソッド ({len(methods)}個):")
                    for i, method in enumerate(methods[:15]):  # 最初の15個を表示
                        print(f"      {i+1:2d}. '{method['method_name']}' (CCN: {method['ccn']}, Length: {method['length']})")

                    if SKIP_MISSING_METHODS:
                        print("  → このメソッドをスキップして続行します")
                        skipped_count += 1
                        continue
                    else:
                        raise ValueError(f"対象メソッド '{method_name}' ({method_type}) がファイル '{java_file_path}' 内で見つかりませんでした")

                # 結果をメトリクスリストに追加
                for method in filtered_methods:
                    all_method_metrics.append({
                        'commit_hash': commit_hash,
                        'original_parent': parent_path,
                        'original_long_name': long_name,
                        'class_name': class_name,
                        'target_method': method['target_method'],
                        'method_type': method['method_type'],
                        'detected_method': method['detected_method'],
                        'ccn': method['ccn'],
                        'length': method['length'],
                        'tokens': method['tokens'],
                        'params': method['params'],
                        'filename': method['filename'],
                        'line_number': method['line_number'],
                        'fallback_used': method.get('fallback_used', False)
                    })

                print(f"  見つかったメソッド数: {len(filtered_methods)}")
                if filtered_methods:
                    print(f"  マッチしたメソッド: {', '.join([m['detected_method'] for m in filtered_methods])}")

        finally:
            # 全ての処理が終了したら元のコミットに戻る
            for repo_path_key, original_head in original_heads.items():
                if original_head:
                    restore_original_head(repo_path_key, original_head)

        # 結果をCSVファイルに保存
        if all_method_metrics:
            metrics_df = pd.DataFrame(all_method_metrics)
            metrics_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n結果が '{output_csv}' に保存されました")
            print(f"総メトリクス数: {len(all_method_metrics)}")
            print(f"スキップされたレコード数: {skipped_count}")

            # フォールバック使用統計
            fallback_count = len(metrics_df[metrics_df['fallback_used'] == True])
            print(f"フォールバック戦略使用数: {fallback_count}")

            # 簡単な統計情報を表示
            print("\n=== 統計情報 ===")
            print(f"平均CCN: {metrics_df['ccn'].mean():.2f}")
            print(f"平均Length: {metrics_df['length'].mean():.2f}")
            print(f"平均Tokens: {metrics_df['tokens'].mean():.2f}")
            print(f"平均Parameters: {metrics_df['params'].mean():.2f}")

            # メソッドタイプ別の統計
            print("\n=== メソッドタイプ別統計 ===")
            type_stats = metrics_df.groupby('method_type').agg({
                'ccn': 'mean',
                'length': 'mean',
                'tokens': 'mean',
                'params': 'mean'
            }).round(2)
            print(type_stats)

        else:
            print("\n警告: 対象メソッドのメトリクスが見つかりませんでした")
            print(f"スキップされたレコード数: {skipped_count}")

    except Exception as e:
        print(f"エラー: 処理中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
