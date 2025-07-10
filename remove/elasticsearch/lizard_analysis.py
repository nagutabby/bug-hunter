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
    例: org.elasticsearch.search.aggregations.InternalOrder$CompoundOrder
    → CompoundOrder (内部クラス名を返す)
    """
    if not parent_path or pd.isna(parent_path):
        return None, None

    # 最後のドット以降を取得
    class_part = parent_path.split('.')[-1]

    # $マークがある場合の処理（内部クラス）
    if '$' in class_part:
        # $で分割して外部クラスと内部クラスを取得
        parts = class_part.split('$')
        outer_class = parts[0]
        inner_class = parts[-1]  # 最後の部分が実際のターゲットクラス

        # 数字で始まる場合（匿名内部クラス）の処理
        if inner_class and inner_class[0].isdigit():
            # 数字以降の部分を取得
            match = re.match(r'^\d+(.+)$', inner_class)
            if match:
                actual_class_name = match.group(1)
                return actual_class_name, outer_class
            else:
                # 数字のみの場合は外部クラス名を使用
                return outer_class, outer_class
        else:
            # 通常の内部クラス
            return inner_class, outer_class

    # 通常のクラス（内部クラスでない）
    return class_part, None

def extract_method_name(long_name):
    """
    LongNameカラムからメソッド名とシグネチャを抽出する（改良版 - シグネチャ対応）
    """
    if not long_name or pd.isna(long_name):
        return None, None, None

    # メソッドシグネチャの括弧の位置を見つける
    paren_index = long_name.find('(')
    if paren_index == -1:
        return None, None, None

    # 括弧の前の部分を取得
    method_part = long_name[:paren_index]

    # 最後のドットの位置を見つける
    last_dot_index = method_part.rfind('.')
    if last_dot_index == -1:
        return None, None, None

    # メソッド名を抽出
    method_name = method_part[last_dot_index + 1:]

    # メソッドシグネチャを抽出
    close_paren_index = long_name.find(')', paren_index)
    if close_paren_index != -1:
        # 引数部分とreturn typeを含むシグネチャ
        signature = long_name[paren_index:]
        # 引数部分のみ
        args_part = long_name[paren_index + 1:close_paren_index]
    else:
        signature = long_name[paren_index:]
        args_part = ""

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
            return actual_class_name, 'constructor', signature
        else:
            # 通常のクラスのコンストラクタ
            return class_name_part, 'constructor', signature

    # <clinit>はスタティックイニシャライザ
    elif method_name == '<clinit>':
        return 'static_initializer', 'method', signature

    # 通常のメソッド
    else:
        return method_name, 'method', signature

def extract_package_path(parent_path):
    """
    Parentカラムからパッケージパスを抽出する
    例: org.elasticsearch.search.aggregations.InternalOrder$CompoundOrder
    → org/elasticsearch/search/aggregations/
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

def get_commit_sequence(repo_path, start_commit_hash, num_commits=10):
    """
    指定されたコミットから始まって、指定された数のコミットの履歴を取得する
    """
    try:
        from git import Repo
        repo = Repo(repo_path)

        # 開始コミットを取得
        start_commit = repo.commit(start_commit_hash)

        # コミット履歴を取得（現在のコミットから遡る）
        commit_list = list(repo.iter_commits(start_commit, max_count=num_commits))

        # コミットハッシュのリストを返す（時系列順に並び替え）
        commit_hashes = [commit.hexsha for commit in reversed(commit_list)]

        print(f"取得したコミット履歴 ({len(commit_hashes)}個):")
        for i, hash_val in enumerate(commit_hashes):
            print(f"  {i+1:2d}. {hash_val}")

        return commit_hashes

    except Exception as e:
        print(f"エラー: コミット履歴の取得中にエラーが発生しました: {e}")
        return []

def checkout_commit(repo_path, commit_hash):
    """
    指定されたコミットにチェックアウトする
    """
    try:
        from git import Repo
        repo = Repo(repo_path)

        # 指定されたコミットにチェックアウト
        repo.git.checkout(commit_hash)

        print(f"成功: コミット {commit_hash} にチェックアウトしました")
        return True

    except Exception as e:
        print(f"エラー: チェックアウト中に例外が発生しました: {e}")
        return False

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

def parse_java_signature_params(args_str):
    """
    Javaのメソッドシグネチャから引数数を正確に解析する
    例:
    - '' → 0 (引数なし)
    - 'I' → 1 (int 1個)
    - 'Ljava/lang/String;I' → 2 (String 1個, int 1個)
    - '[I' → 1 (int配列 1個)
    """
    if not args_str:
        return 0

    param_count = 0
    i = 0

    while i < len(args_str):
        char = args_str[i]

        if char in 'ZBCSIJFD':
            # プリミティブ型（boolean, byte, char, short, int, long, float, double）
            param_count += 1
            i += 1
        elif char == 'L':
            # オブジェクト型（Ljava/lang/String; など）
            param_count += 1
            # セミコロンまでスキップ
            while i < len(args_str) and args_str[i] != ';':
                i += 1
            i += 1  # セミコロンもスキップ
        elif char == '[':
            # 配列型
            i += 1  # '[' をスキップ
            # 配列の要素型を処理（再帰的に処理するが、カウントは1つ）
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
            # 未知の文字は無視
            i += 1

    return param_count

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

def filter_methods_by_target(methods, target_method_name, target_class_name, target_signature=None, outer_class_name=None, method_type='method'):
    """
    対象のメソッド名、クラス名、シグネチャでフィルタリングする（改良版 - 重複防止対応）
    """
    filtered_methods = []

    for method in methods:
        method_name = method['method_name']
        matched = False

        if method_type == 'constructor':
            # コンストラクタの場合の厳密なマッチング
            constructor_patterns = [
                # 完全一致パターン
                target_method_name,
                f"{target_class_name}::{target_method_name}",
            ]

            # 内部クラスの場合の追加パターン
            if outer_class_name:
                constructor_patterns.extend([
                    f"{outer_class_name}::{target_method_name}",
                    f"{outer_class_name}${target_class_name}::{target_method_name}",
                ])

            # パターンマッチング
            for pattern in constructor_patterns:
                if method_name == pattern:
                    matched = True
                    break

            # 追加の厳密チェック
            if not matched:
                # method_nameが "クラス名::クラス名" の形式の場合
                if '::' in method_name:
                    parts = method_name.split('::')
                    if len(parts) == 2:
                        class_part = parts[0]
                        method_part = parts[1]

                        # ターゲットクラス名とマッチするかチェック
                        if (method_part == target_method_name and
                            (class_part == target_class_name or
                             class_part.endswith(f"${target_class_name}"))):
                            matched = True
                elif method_name == target_method_name:
                    matched = True

        else:
            # 通常のメソッドの場合
            if method_name == target_method_name:
                matched = True
            elif '::' in method_name:
                parts = method_name.split('::')
                if len(parts) == 2:
                    class_part = parts[0]
                    method_part = parts[1]

                    # メソッド名が一致する場合
                    if method_part == target_method_name:
                        # クラス名も正確にマッチするかチェック
                        if target_class_name and outer_class_name:
                            # 内部クラスの場合: クラス名が正確にマッチするかチェック
                            if (class_part == target_class_name or
                                class_part.endswith(f"${target_class_name}") or
                                class_part == f"{outer_class_name}${target_class_name}"):
                                matched = True
                        elif target_class_name:
                            # 通常のクラスの場合
                            if (class_part == target_class_name or
                                class_part.endswith(f"${target_class_name}")):
                                matched = True
                        else:
                            # クラス名が指定されていない場合はメソッド名のみでマッチ
                            matched = True

        # シグネチャでの追加フィルタリング（オーバーロード対応）
        if matched and target_signature:
            # 引数部分を抽出
            paren_end = target_signature.find(')')
            if paren_end != -1:
                args_str = target_signature[1:paren_end]  # ()の中身
                return_type = target_signature[paren_end + 1:] if paren_end + 1 < len(target_signature) else ""

                # 引数数を正確に計算
                expected_params = parse_java_signature_params(args_str)

                # 引数数が一致しない場合はマッチしない
                if method['params'] != expected_params:
                    matched = False
                    continue

                # 同じ引数数の場合は、既に追加済みかチェックして重複を防ぐ
                if matched:
                    # 既に同じターゲットの結果が追加されているかチェック
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

def analyze_with_lizard_only(java_file_path, target_method_name, target_class_name, target_signature=None, outer_class_name=None, method_type='method', debug=False):
    """
    Lizardのみでメソッドを検索する（フォールバック処理なし）
    """
    # Lizard分析
    methods = analyze_java_file_with_lizard(java_file_path)

    if debug:
        print(f"    Lizardで検出された全メソッド ({len(methods)}個):")
        for i, method in enumerate(methods):
            print(f"      {i+1:2d}. '{method['method_name']}' (CCN: {method['ccn']}, Params: {method['params']})")

    # フィルタリング
    filtered_methods = filter_methods_by_target(
        methods, target_method_name, target_class_name, target_signature, outer_class_name, method_type
    )

    if debug:
        print(f"    フィルタリング後のメソッド ({len(filtered_methods)}個):")
        for method in filtered_methods:
            print(f"      - '{method['detected_method']}' -> '{method['target_method']}' (クラス: {method['target_class']}, Params: {method['params']})")

    if filtered_methods:
        return filtered_methods, "Lizard分析成功"
    else:
        return [], "Lizardでメソッドが見つからない"

def track_method_complexity_changes(repo_path, start_commit_hash, parent_path, long_name, num_commits=10, debug=False):
    """
    指定されたコミットから始まって複数のコミットでメソッドの複雑度変化を追跡する
    """
    print(f"\n=== メソッド複雑度変化追跡開始 ===")
    print(f"開始コミット: {start_commit_hash}")
    print(f"Parent: {parent_path}")
    print(f"LongName: {long_name}")
    print(f"追跡コミット数: {num_commits}")

    try:
        # コミット履歴を取得
        commit_sequence = get_commit_sequence(repo_path, start_commit_hash, num_commits)
        if not commit_sequence:
            print("エラー: コミット履歴を取得できませんでした")
            return []

        # メソッド情報を抽出
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

        complexity_data = []

        # 各コミットで複雑度を測定
        for i, commit_hash in enumerate(commit_sequence):
            print(f"\n--- コミット {i+1}/{len(commit_sequence)}: {commit_hash} ---")

            # コミットにチェックアウト
            success = checkout_commit(repo_path, commit_hash)
            if not success:
                print(f"  スキップ: チェックアウトに失敗")
                continue

            # Javaファイルを検索
            java_file_path = find_java_file_in_filesystem(repo_path, search_class_name, package_path)
            if not java_file_path:
                print(f"  スキップ: Javaファイルが見つかりません (クラス: {search_class_name})")
                continue

            print(f"  見つかったファイル: {java_file_path}")

            # メソッドを分析（Lizardのみ）
            filtered_methods, strategy = analyze_with_lizard_only(
                java_file_path,
                method_name,
                target_class_name,
                method_signature,
                outer_class_name,
                method_type,
                debug=debug
            )

            if len(filtered_methods) == 0:
                print(f"  スキップ: Lizardでメソッドが見つかりませんでした ({strategy})")
                continue
            elif len(filtered_methods) > 1:
                print(f"  警告: 複数のメソッドがマッチしました ({len(filtered_methods)}個) - 最初のものを使用")

            # 複雑度データを記録
            method_data = filtered_methods[0]
            complexity_data.append({
                'commit_order': i + 1,
                'commit_hash': commit_hash,
                'ccn': method_data['ccn'],
                'length': method_data['length'],
                'tokens': method_data['tokens'],
                'params': method_data['params'],
                'filename': method_data['filename'],
                'line_number': method_data['line_number'],
                'strategy': strategy
            })

            # 結果表示
            print(f"  複雑度 (CCN): {method_data['ccn']}")
            print(f"  長さ: {method_data['length']}")
            print(f"  トークン数: {method_data['tokens']}")
            print(f"  パラメーター数: {method_data['params']}")

        return complexity_data

    except Exception as e:
        print(f"エラー: 複雑度追跡中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_complexity_statistics(complexity_data):
    """
    複雑度データから統計情報を計算する
    """
    if len(complexity_data) < 2:
        print("統計計算には最低2つのデータポイントが必要です")
        return None

    # 各メトリックの値を抽出
    ccn_values = [data['ccn'] for data in complexity_data if data['ccn'] is not None]
    length_values = [data['length'] for data in complexity_data if data['length'] is not None]
    tokens_values = [data['tokens'] for data in complexity_data if data['tokens'] is not None]
    params_values = [data['params'] for data in complexity_data if data['params'] is not None]

    # 有効なデータポイントが不足している場合
    if len(ccn_values) < 2:
        print("統計計算には最低2つの有効なCCN値が必要です")
        return None

    # 標準偏差を計算
    import numpy as np
    ccn_std = np.std(ccn_values) if len(ccn_values) >= 2 else None
    length_std = np.std(length_values) if len(length_values) >= 2 else None
    tokens_std = np.std(tokens_values) if len(tokens_values) >= 2 else None
    params_std = np.std(params_values) if len(params_values) >= 2 else None

    # 統計情報
    stats = {
        'data_points': len(complexity_data),
        'valid_ccn_points': len(ccn_values),
        'valid_length_points': len(length_values),
        'valid_tokens_points': len(tokens_values),
        'valid_params_points': len(params_values),
        'ccn_std': ccn_std,
        'length_std': length_std,
        'tokens_std': tokens_std,
        'params_std': params_std,
        'ccn_values': ccn_values,
        'length_values': length_values,
        'tokens_values': tokens_values,
        'params_values': params_values
    }

    return stats

def prepare_enhanced_csv_output(original_df, complexity_results):
    """
    元のCSVデータに複雑度統計の新しいカラムを追加したデータフレームを作成する
    """
    # 元のDataFrameをコピー
    enhanced_df = original_df.copy()

    # 新しいカラムを初期化（NaN値で）
    new_columns = [
        'tracking_data_points',
        'tracking_ccn_std',
        'tracking_length_std',
        'tracking_tokens_std',
        'tracking_params_std',
    ]

    for col in new_columns:
        enhanced_df[col] = pd.NA

    # 複雑度結果を元のデータフレームにマージ
    for record_id, stats in complexity_results.items():
        row_index = record_id - 1  # record_idは1ベース、DataFrameは0ベース

        if stats is not None:
            # 処理に成功した場合のみデータを設定
            enhanced_df.loc[row_index, 'tracking_data_points'] = stats['data_points']
            enhanced_df.loc[row_index, 'tracking_ccn_std'] = stats['ccn_std']
            enhanced_df.loc[row_index, 'tracking_length_std'] = stats['length_std']
            enhanced_df.loc[row_index, 'tracking_tokens_std'] = stats['tokens_std']
            enhanced_df.loc[row_index, 'tracking_params_std'] = stats['params_std']

    return enhanced_df

def main():
    # 設定
    csv_file = "method-p_filtered_v2.csv"
    repo_path = "/Users/nagutabby/elasticsearch"
    enhanced_output_csv = "method-p_filtered_v2_enhanced.csv"  # 拡張データ用の新しいファイル名
    num_commits = 30  # 追跡するコミット数
    max_records = 2000  # 処理する最大レコード数

    # 処理をスキップするかどうかのフラグ
    SKIP_MISSING_METHODS = True  # メソッドが見つからない場合はスキップ
    DEBUG_MODE = False  # デバッグ情報を表示

    print("=== Git Repository Analysis Tool ===")
    print("注意: このスクリプトはリポジトリの状態を変更します。")
    print("分析後、リポジトリは最後に処理されたコミットの状態になります。")
    print("分析前に必要な作業をコミット・保存してください。")

    response = input("\n続行しますか？ (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("処理を中止しました。")
        sys.exit(0)

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

        # 最初の2000レコードのみを処理対象とする
        if len(df) > max_records:
            print(f"情報: CSVファイルには{len(df)}レコードありますが、最初の{max_records}レコードのみ処理します")
            df_to_process = df.head(max_records)
        else:
            print(f"情報: CSVファイルの全{len(df)}レコードを処理します")
            df_to_process = df

        all_tracking_results = []
        processed_count = 0
        skipped_count = 0
        complexity_results = {}  # record_id -> stats のマッピング

        print(f"\n{'='*80}")
        print(f"処理開始: {len(df_to_process)}レコードを処理します")
        print(f"{'='*80}")

        # 各レコードを処理
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
                # メソッドの複雑度変化を追跡
                complexity_data = track_method_complexity_changes(
                    repo_path,
                    commit_hash,
                    parent_path,
                    long_name,
                    num_commits,
                    debug=DEBUG_MODE
                )

                if len(complexity_data) == 0:
                    print(f"  結果: データが取得できませんでした")
                    complexity_results[record_id] = None
                    if SKIP_MISSING_METHODS:
                        skipped_count += 1
                        continue
                    else:
                        raise ValueError("メソッドの複雑度データを取得できませんでした")

                # 統計情報を計算
                stats = calculate_complexity_statistics(complexity_data)

                if stats:
                    print(f"\n=== 統計結果 ===")
                    print(f"データポイント数: {stats['data_points']}")

                    print(f"CCN標準偏差: {stats['ccn_std']:.3f}")
                    print(f"長さ標準偏差: {stats['length_std']:.3f}")
                    print(f"トークン標準偏差: {stats['tokens_std']:.3f}")
                    print(f"パラメーター標準偏差: {stats['params_std']:.3f}")

                    # 統計結果を保存
                    complexity_results[record_id] = stats

                    processed_count += 1

                else:
                    print(f"  結果: 統計情報を計算できませんでした")
                    complexity_results[record_id] = None
                    if SKIP_MISSING_METHODS:
                        skipped_count += 1
                        continue

            except Exception as e:
                print(f"エラー: レコード {record_id} の処理中に例外が発生しました: {e}")
                complexity_results[record_id] = None
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                if SKIP_MISSING_METHODS:
                    skipped_count += 1
                    continue
                else:
                    raise

        # 拡張版CSVを作成して保存
        enhanced_df = prepare_enhanced_csv_output(df, complexity_results)
        enhanced_df.to_csv(enhanced_output_csv, index=False, encoding='utf-8')
        print(f"\n{'='*80}")
        print(f"拡張版データが '{enhanced_output_csv}' に保存されました")

        # 結果サマリーを表示
        print(f"\n{'='*80}")
        print(f"=== 処理完了サマリー ===")
        print(f"対象レコード数: {len(df_to_process)}")
        print(f"処理されたレコード数: {processed_count}")
        print(f"スキップされたレコード数: {skipped_count}")
        print(f"成功率: {processed_count/len(df_to_process)*100:.1f}%")

        # 全体の統計情報を表示
        if processed_count > 0:
            print(f"\n=== 全体統計 ===")
            # 各レコードごとの標準偏差の統計
            if complexity_results:
                # 成功したレコードの統計データを取得
                successful_stats = [stats for stats in complexity_results.values() if stats is not None]

                if successful_stats:
                    ccn_stds = [stats['ccn_std'] for stats in successful_stats if stats['ccn_std'] is not None]
                    length_stds = [stats['length_std'] for stats in successful_stats if stats['length_std'] is not None]
                    tokens_stds = [stats['tokens_std'] for stats in successful_stats if stats['tokens_std'] is not None]
                    params_stds = [stats['params_std'] for stats in successful_stats if stats['params_std'] is not None]

                    import numpy as np

                    if ccn_stds:
                        print(f"CCN標準偏差:")
                        print(f"  平均: {np.mean(ccn_stds):.3f}")
                        print(f"  中央値: {np.median(ccn_stds):.3f}")
                        print(f"  標準偏差: {np.std(ccn_stds):.3f}")

                    if length_stds:
                        print(f"\n長さ標準偏差:")
                        print(f"  平均: {np.mean(length_stds):.3f}")
                        print(f"  中央値: {np.median(length_stds):.3f}")
                        print(f"  標準偏差: {np.std(length_stds):.3f}")

                    if tokens_stds:
                        print(f"\nトークン標準偏差:")
                        print(f"  平均: {np.mean(tokens_stds):.3f}")
                        print(f"  中央値: {np.median(tokens_stds):.3f}")
                        print(f"  標準偏差: {np.std(tokens_stds):.3f}")

                    if params_stds:
                        print(f"\nパラメーター標準偏差:")
                        print(f"  平均: {np.mean(params_stds):.3f}")
                        print(f"  中央値: {np.median(params_stds):.3f}")
                        print(f"  標準偏差: {np.std(params_stds):.3f}")

        else:
            print("\n警告: 処理できたデータがありませんでした")
            print(f"スキップされたレコード数: {skipped_count}")

        print(f"\n{'='*80}")
        print("処理が正常に完了しました。")
        print(f"出力ファイル: {enhanced_output_csv}")
        print("\n注意: リポジトリは最後に処理されたコミットの状態になっています。")
        print("必要に応じて手動で元のブランチ/コミットに戻してください。")

    except Exception as e:
        print(f"エラー: メイン処理中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
