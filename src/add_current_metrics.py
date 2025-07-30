import pandas as pd
import os
import sys
import re
import glob
from pathlib import Path
from lizard import analyze_file

def extract_class_name(parent_path):
    if not parent_path or pd.isna(parent_path):
        return None, None

    class_part = parent_path.split('.')[-1]

    if '$' in class_part:
        parts = class_part.split('$')
        outer_class = parts[0]
        inner_class = parts[-1]

        if inner_class and inner_class[0].isdigit():
            match = re.match(r'^\d+(.+)$', inner_class)
            if match:
                actual_class_name = match.group(1)
                return actual_class_name, outer_class
            else:
                return outer_class, outer_class
        else:
            return inner_class, outer_class

    return class_part, None

def extract_method_name(long_name):
    if not long_name or pd.isna(long_name):
        return None, None, None

    paren_index = long_name.find('(')
    if paren_index == -1:
        return None, None, None

    method_part = long_name[:paren_index]
    last_dot_index = method_part.rfind('.')
    if last_dot_index == -1:
        return None, None, None

    method_name = method_part[last_dot_index + 1:]
    close_paren_index = long_name.find(')', paren_index)
    if close_paren_index != -1:
        signature = long_name[paren_index:]
        args_part = long_name[paren_index + 1:close_paren_index]
    else:
        signature = long_name[paren_index:]
        args_part = ""

    class_part = method_part[:last_dot_index]
    class_name_part = class_part.split('.')[-1] if '.' in class_part else class_part

    if method_name == '<init>':
        if '$' in class_name_part:
            class_parts = class_name_part.split('$')
            inner_class_part = class_parts[-1]

            if inner_class_part and inner_class_part[0].isdigit():
                match = re.match(r'^\d+(.+)$', inner_class_part)
                if match:
                    actual_class_name = match.group(1)
                else:
                    actual_class_name = class_parts[0] if len(class_parts) > 1 else inner_class_part
            else:
                actual_class_name = inner_class_part

            return actual_class_name, 'constructor', signature
        else:
            return class_name_part, 'constructor', signature
    elif method_name == '<clinit>':
        return 'static_initializer', 'method', signature
    else:
        return method_name, 'method', signature

def extract_package_path(parent_path):
    if not parent_path or pd.isna(parent_path):
        return None

    parts = parent_path.split('.')
    if len(parts) <= 1:
        return None

    package_parts = parts[:-1]
    return '/'.join(package_parts) + '/'

def get_current_commit(repo_path, commit_hash):
    try:
        from git import Repo
        repo = Repo(repo_path)
        commit = repo.commit(commit_hash)
        print(f"対象コミット: {commit_hash}")
        return [commit_hash], ["現在のコミット"]
    except Exception as e:
        print(f"エラー: コミット取得中にエラーが発生しました: {e}")
        return [], []

def checkout_commit(repo_path, commit_hash):
    try:
        from git import Repo
        repo = Repo(repo_path)
        repo.git.checkout(commit_hash)
        print(f"成功: コミット {commit_hash} にチェックアウトしました")
        return True
    except Exception as e:
        print(f"エラー: チェックアウト中に例外が発生しました: {e}")
        return False

def find_java_file_in_filesystem(repo_path, class_name, package_path):
    try:
        search_patterns = []

        if package_path:
            pattern1 = os.path.join(repo_path, "**", package_path, f"{class_name}.java")
            search_patterns.append(pattern1)
            pattern2 = os.path.join(repo_path, "**", "src", "main", "java", package_path, f"{class_name}.java")
            search_patterns.append(pattern2)
            pattern3 = os.path.join(repo_path, "**", "src", "test", "java", package_path, f"{class_name}.java")
            search_patterns.append(pattern3)

        pattern4 = os.path.join(repo_path, "**", f"{class_name}.java")
        search_patterns.append(pattern4)

        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]

        return None
    except Exception as e:
        print(f"エラー: ファイルシステム検索中に例外が発生しました: {e}")
        return None

def parse_java_signature_params(args_str):
    if not args_str:
        return 0

    param_count = 0
    i = 0

    while i < len(args_str):
        char = args_str[i]

        if char in 'ZBCSIJFD':
            param_count += 1
            i += 1
        elif char == 'L':
            param_count += 1
            while i < len(args_str) and args_str[i] != ';':
                i += 1
            i += 1
        elif char == '[':
            i += 1
            if i < len(args_str):
                if args_str[i] in 'ZBCSIJFD':
                    param_count += 1
                    i += 1
                elif args_str[i] == 'L':
                    param_count += 1
                    while i < len(args_str) and args_str[i] != ';':
                        i += 1
                    i += 1
        else:
            i += 1

    return param_count

def analyze_java_file_with_lizard(file_path):
    try:
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

def filter_methods_by_target(methods, target_method_name, target_class_name, target_signature=None, outer_class_name=None, method_type='method'):
    filtered_methods = []

    for method in methods:
        method_name = method['method_name']
        matched = False

        if method_type == 'constructor':
            constructor_patterns = [
                target_method_name,
                f"{target_class_name}::{target_method_name}",
            ]

            if outer_class_name:
                constructor_patterns.extend([
                    f"{outer_class_name}::{target_method_name}",
                    f"{outer_class_name}${target_class_name}::{target_method_name}",
                ])

            for pattern in constructor_patterns:
                if method_name == pattern:
                    matched = True
                    break

            if not matched:
                if '::' in method_name:
                    parts = method_name.split('::')
                    if len(parts) == 2:
                        class_part = parts[0]
                        method_part = parts[1]

                        if (method_part == target_method_name and
                            (class_part == target_class_name or
                             class_part.endswith(f"${target_class_name}"))):
                            matched = True
                elif method_name == target_method_name:
                    matched = True

        else:
            if method_name == target_method_name:
                matched = True
            elif '::' in method_name:
                parts = method_name.split('::')
                if len(parts) == 2:
                    class_part = parts[0]
                    method_part = parts[1]

                    if method_part == target_method_name:
                        if target_class_name and outer_class_name:
                            if (class_part == target_class_name or
                                class_part.endswith(f"${target_class_name}") or
                                class_part == f"{outer_class_name}${target_class_name}"):
                                matched = True
                        elif target_class_name:
                            if (class_part == target_class_name or
                                class_part.endswith(f"${target_class_name}")):
                                matched = True
                        else:
                            matched = True

        if matched and target_signature:
            paren_end = target_signature.find(')')
            if paren_end != -1:
                args_str = target_signature[1:paren_end]
                expected_params = parse_java_signature_params(args_str)

                if method['params'] != expected_params:
                    matched = False
                    continue

                if matched:
                    already_added = any(
                        fm['target_method'] == target_method_name and
                        fm['target_class'] == target_class_name and
                        fm['params'] == method['params'] and
                        fm.get('target_signature') == target_signature
                        for fm in filtered_methods
                    )
                    if already_added:
                        matched = False

        if matched:
            filtered_methods.append({
                **method,
                'target_method': target_method_name,
                'target_class': target_class_name,
                'target_signature': target_signature,
                'outer_class': outer_class_name,
                'method_type': method_type,
                'detected_method': method_name
            })

    return filtered_methods

def analyze_current_state_only(java_file_path, target_method_name, target_class_name, target_signature=None, outer_class_name=None, method_type='method', debug=False):
    methods = analyze_java_file_with_lizard(java_file_path)

    if debug:
        print(f"    Lizardで検出された全メソッド ({len(methods)}個):")
        for i, method in enumerate(methods):
            print(f"      {i+1:2d}. '{method['method_name']}' (CCN: {method['ccn']}, Params: {method['params']})")

    filtered_methods = filter_methods_by_target(
        methods, target_method_name, target_class_name, target_signature, outer_class_name, method_type
    )

    if debug:
        print(f"    フィルタリング後のメソッド ({len(filtered_methods)}個):")
        for method in filtered_methods:
            print(f"      - '{method['detected_method']}' -> '{method['target_method']}' (クラス: {method['target_class']}, Params: {method['params']})")

    if filtered_methods:
        return filtered_methods, "現在状態分析成功"
    else:
        return [], "現在状態でメソッドが見つからない"

def track_current_state(repo_path, commit_hash, parent_path, long_name, debug=False):
    print(f"\n=== 現在状態記録開始 ===")
    print(f"対象コミット: {commit_hash}")
    print(f"Parent: {parent_path}")
    print(f"LongName: {long_name}")

    try:
        target_class_name, outer_class_name = extract_class_name(parent_path)
        method_name, method_type, method_signature = extract_method_name(long_name)
        package_path = extract_package_path(parent_path)
        search_class_name = outer_class_name if outer_class_name else target_class_name

        print(f"\n対象メソッド情報:")
        print(f"  ターゲットクラス名: {target_class_name}")
        print(f"  外部クラス名: {outer_class_name}")
        print(f"  メソッド名: {method_name} ({method_type})")
        print(f"  メソッドシグネチャ: {method_signature}")
        print(f"  パッケージパス: {package_path}")
        print(f"  検索用クラス名: {search_class_name}")

        success = checkout_commit(repo_path, commit_hash)
        if not success:
            print(f"  スキップ: チェックアウトに失敗")
            return None

        java_file_path = find_java_file_in_filesystem(repo_path, search_class_name, package_path)
        if not java_file_path:
            print(f"  スキップ: Javaファイルが見つかりません (クラス: {search_class_name})")
            return None

        print(f"  見つかったファイル: {java_file_path}")

        filtered_methods, strategy = analyze_current_state_only(
            java_file_path,
            method_name,
            target_class_name,
            method_signature,
            outer_class_name,
            method_type,
            debug=debug
        )

        if len(filtered_methods) == 0:
            print(f"  結果: メソッドが見つかりませんでした ({strategy})")
            return None
        elif len(filtered_methods) > 1:
            print(f"  警告: 複数のメソッドがマッチしました ({len(filtered_methods)}個) - 最初のものを使用")

        # 現在状態データを記録
        method_data = filtered_methods[0]
        current_state = {
            'commit_hash': commit_hash,
            'current_ccn': method_data['ccn'],
            'current_length': method_data['length'],
            'current_tokens': method_data['tokens'],
            'filename': method_data['filename'],
            'line_number': method_data['line_number'],
            'strategy': strategy
        }

        print(f"  現在のCCN: {method_data['ccn']}")
        print(f"  現在の長さ: {method_data['length']}")
        print(f"  現在のトークン数: {method_data['tokens']}")

        return current_state

    except Exception as e:
        print(f"エラー: 現在状態記録中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_current_state_csv_output(original_df, current_state_results):
    enhanced_df = original_df.copy()

    new_columns = [
        'current_ccn',
        'current_length',
        'current_tokens',
    ]

    for col in new_columns:
        enhanced_df[col] = pd.NA

    for record_id, state in current_state_results.items():
        row_index = record_id - 1

        if state is not None:
            enhanced_df.loc[row_index, 'current_ccn'] = state['current_ccn']
            enhanced_df.loc[row_index, 'current_length'] = state['current_length']
            enhanced_df.loc[row_index, 'current_tokens'] = state['current_tokens']

    return enhanced_df

def main():
    base_dir = "../data/remove/ceylon-ide-eclipse/"
    input_path = base_dir + "method-p_drop_columns_rows.csv"
    output_path = base_dir + "method-p_add_current_metrics.csv"
    repo_path = "/Users/nagutabby/ceylon-ide-eclipse"
    max_records = 3000

    SKIP_MISSING_METHODS = True
    DEBUG_MODE = False

    print("=== Git Repository Analysis Tool (現在状態記録版) ===")
    print("注意: このスクリプトはリポジトリの状態を変更します。")
    print("分析後、リポジトリは最後に処理されたコミットの状態になります。")
    print("分析前に必要な作業をコミット・保存してください。")
    print("\nこのバージョンでは各メソッドについて:")
    print("- 指定されたコミットでの現在状態を記録")
    print("- CCN、長さ、トークン数の絶対値を記録")

    response = input("\n続行しますか？ (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("処理を中止しました。")
        sys.exit(0)

    if not os.path.exists(input_path):
        print(f"エラー: CSVファイル '{input_path}' が見つかりません")
        sys.exit(1)

    if not os.path.exists(repo_path):
        print(f"エラー: リポジトリパス '{repo_path}' が見つかりません")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)

        required_columns = ['Hash', 'Parent', 'LongName']
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

        processed_count = 0
        skipped_count = 0
        current_state_results = {}

        print(f"\n{'='*80}")
        print(f"処理開始: {len(df_to_process)}レコードを処理します")
        print(f"{'='*80}")

        for idx, record in df_to_process.iterrows():
            commit_hash = record['Hash']
            parent_path = record['Parent']
            long_name = record['LongName']
            record_id = idx + 1

            print(f"\n{'='*80}")
            print(f"処理中のレコード {record_id}/{len(df_to_process)}:")
            print(f"  Hash: {commit_hash}")
            print(f"  Parent: {parent_path}")
            print(f"  LongName: {long_name}")

            try:
                current_state = track_current_state(
                    repo_path,
                    commit_hash,
                    parent_path,
                    long_name,
                    debug=DEBUG_MODE
                )

                print(f"\n=== 分析結果 ===")
                if current_state is not None:
                    print(f"現在のCCN: {current_state['current_ccn']}")
                    print(f"現在の長さ: {current_state['current_length']}")
                    print(f"現在のトークン数: {current_state['current_tokens']}")
                else:
                    print(f"メソッドの現在状態を取得できませんでした")

                current_state_results[record_id] = current_state
                processed_count += 1

            except Exception as e:
                print(f"エラー: レコード {record_id} の処理中に例外が発生しました: {e}")
                current_state_results[record_id] = None
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                if SKIP_MISSING_METHODS:
                    skipped_count += 1
                    continue
                else:
                    raise

        enhanced_df = prepare_current_state_csv_output(df, current_state_results)
        enhanced_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n{'='*80}")
        print(f"現在状態データが '{output_path}' に保存されました")

        print(f"\n{'='*80}")
        print(f"=== 処理完了サマリー ===")
        print(f"対象レコード数: {len(df_to_process)}")
        print(f"処理されたレコード数: {processed_count}")
        print(f"スキップされたレコード数: {skipped_count}")
        print(f"成功率: {processed_count/len(df_to_process)*100:.1f}%")

        print(f"\n{'='*80}")
        print("処理が正常に完了しました。")
        print(f"出力ファイル: {output_path}")
        print("\n注意: リポジトリは最後に処理されたコミットの状態になっています。")
        print("必要に応じて手動で元のブランチ/コミットに戻してください。")

    except Exception as e:
        print(f"エラー: メイン処理中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
