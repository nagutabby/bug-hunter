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

def get_current_and_previous_commit(repo_path, start_commit_hash):
    """
    指定されたコミットとその1つ前のコミットを取得する
    """
    try:
        from git import Repo
        repo = Repo(repo_path)

        # 開始コミットを取得
        start_commit = repo.commit(start_commit_hash)

        # 現在のコミットとその1つ前のコミットを取得
        commit_list = list(repo.iter_commits(start_commit, max_count=2))

        if len(commit_list) < 2:
            print(f"警告: コミット {start_commit_hash} の前のコミットが見つかりません")
            return [start_commit_hash], ["現在のコミットのみ"]

        # コミットハッシュのリストを返す（時系列順に並び替え: [前のコミット, 現在のコミット]）
        current_commit = commit_list[0].hexsha
        previous_commit = commit_list[1].hexsha

        commit_hashes = [previous_commit, current_commit]
        commit_labels = ["前のコミット", "現在のコミット"]

        print(f"取得したコミット:")
        print(f"  前のコミット: {previous_commit}")
        print(f"  現在のコミット: {current_commit}")

        return commit_hashes, commit_labels

    except Exception as e:
        print(f"エラー: コミット履歴の取得中にエラーが発生しました: {e}")
        return [], []

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

def track_method_complexity_changes(repo_path, start_commit_hash, parent_path, long_name, debug=False):
    """
    指定されたコミットとその1つ前のコミットでメソッドの複雑度変化を追跡する
    各コミットでメソッドが見つからない場合も適切にハンドリングする
    """
    print(f"\n=== メソッド複雑度変化追跡開始 ===")
    print(f"開始コミット: {start_commit_hash}")
    print(f"Parent: {parent_path}")
    print(f"LongName: {long_name}")

    try:
        # 現在のコミットとその1つ前のコミットを取得
        commit_sequence, commit_labels = get_current_and_previous_commit(repo_path, start_commit_hash)
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
        for i, (commit_hash, label) in enumerate(zip(commit_sequence, commit_labels)):
            print(f"\n--- {label}: {commit_hash} ---")

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
                print(f"  結果: Lizardでメソッドが見つかりませんでした ({strategy})")
                # メソッドが見つからない場合でも継続（操作タイプの判定のため）
                continue
            elif len(filtered_methods) > 1:
                print(f"  警告: 複数のメソッドがマッチしました ({len(filtered_methods)}個) - 最初のものを使用")

            # 複雑度データを記録
            method_data = filtered_methods[0]
            complexity_data.append({
                'commit_order': i + 1,
                'commit_hash': commit_hash,
                'commit_label': label,
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

        print(f"\n取得されたデータポイント数: {len(complexity_data)}")
        return complexity_data

    except Exception as e:
        print(f"エラー: 複雑度追跡中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_complexity_changes(complexity_data):
    """
    前のコミットと現在のコミットの間の変化量を計算する
    同時に操作の種類（modified/added/deleted/NaN）も判定する

    改良版: added/deletedの場合も変化量を計算する
    - added: 0 → 現在の値への変化
    - deleted: 前の値 → 0への変化
    """
    if len(complexity_data) == 0:
        # パターン1: 両方のコミットでメソッドが見つからない
        return {
            'current_commit': None,
            'ccn_change': None,
            'length_change': None,
            'tokens_change': None,
            'operation_type': 'NaN'
        }
    elif len(complexity_data) == 1:
        # パターン2 or 3: どちらか一方のコミットでのみメソッドが見つかった
        single_data = complexity_data[0]

        if single_data['commit_order'] == 1:
            # パターン3: 前のコミットでのみ見つかった → メソッドが削除された
            # 変化量 = 0（現在の値） - 前の値
            return {
                'current_commit': single_data['commit_hash'],
                'ccn_change': 0 - single_data['ccn'],
                'length_change': 0 - single_data['length'],
                'tokens_change': 0 - single_data['tokens'],
                'operation_type': 'deleted'
            }
        else:
            # パターン2: 現在のコミットでのみ見つかった → メソッドが追加された
            # 変化量 = 現在の値 - 0（前の値）
            return {
                'current_commit': single_data['commit_hash'],
                'ccn_change': single_data['ccn'] - 0,
                'length_change': single_data['length'] - 0,
                'tokens_change': single_data['tokens'] - 0,
                'operation_type': 'added'
            }
    elif len(complexity_data) == 2:
        # パターン4: 両方のコミットでメソッドが見つかった → メソッドが変更された
        previous_data = complexity_data[0]
        current_data = complexity_data[1]

        # 変化量を計算（CCN、長さ、トークンのみ）
        ccn_change = current_data['ccn'] - previous_data['ccn']
        length_change = current_data['length'] - previous_data['length']
        tokens_change = current_data['tokens'] - previous_data['tokens']

        return {
            'current_commit': current_data['commit_hash'],
            'ccn_change': ccn_change,
            'length_change': length_change,
            'tokens_change': tokens_change,
            'operation_type': 'modified'
        }
    else:
        print(f"予期しないデータ数: {len(complexity_data)}")
        return {
            'current_commit': None,
            'ccn_change': None,
            'length_change': None,
            'tokens_change': None,
            'operation_type': 'NaN'
        }

def prepare_enhanced_csv_output(original_df, complexity_results):
    """
    元のCSVデータに複雑度変化の新しいカラムを追加したデータフレームを作成する
    """
    # 元のDataFrameをコピー
    enhanced_df = original_df.copy()

    # 新しいカラムを初期化（NaN値で）
    new_columns = [
        'ccn_change',
        'length_change',
        'tokens_change',
        'operation_type'
    ]

    for col in new_columns:
        enhanced_df[col] = pd.NA

    # 複雑度結果を元のデータフレームにマージ
    for record_id, changes in complexity_results.items():
        row_index = record_id - 1  # record_idは1ベース、DataFrameは0ベース

        if changes is not None:
            # 処理に成功した場合のみデータを設定
            enhanced_df.loc[row_index, 'ccn_change'] = changes['ccn_change']
            enhanced_df.loc[row_index, 'length_change'] = changes['length_change']
            enhanced_df.loc[row_index, 'tokens_change'] = changes['tokens_change']
            enhanced_df.loc[row_index, 'operation_type'] = changes['operation_type']

    return enhanced_df

def main():
    # 設定
    csv_file = "method-p_filtered_v2.csv"
    repo_path = "/Users/nagutabby/elasticsearch"
    enhanced_output_csv = "method-p_filtered_v2_changes_real_number.csv"  # 変化量データ用の新しいファイル名
    max_records = 3000  # 処理する最大レコード数

    # 処理をスキップするかどうかのフラグ
    SKIP_MISSING_METHODS = True  # メソッドが見つからない場合はスキップ
    DEBUG_MODE = False  # デバッグ情報を表示

    print("=== Git Repository Analysis Tool (変化量追跡版) ===")
    print("注意: このスクリプトはリポジトリの状態を変更します。")
    print("分析後、リポジトリは最後に処理されたコミットの状態になります。")
    print("分析前に必要な作業をコミット・保存してください。")
    print("\nこのバージョンでは各メソッドについて:")
    print("- 現在のコミットとその1つ前のコミットを比較")
    print("- CCN、長さ、トークン数の変化量を計算")

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
        complexity_results = {}  # record_id -> changes のマッピング

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
                    debug=DEBUG_MODE
                )

                if complexity_data:
                    # データが取得できた場合の処理は前述の通り
                    pass
                else:
                    print(f"  結果: メソッドのメトリクスが取得できませんでした")
                    # データが取得できない場合でも処理を継続（operation_typeを判定するため）

                # 変化量を計算
                changes = calculate_complexity_changes(complexity_data)

                # 結果を常に表示（operation_typeを含む）
                print(f"\n=== 分析結果 ===")
                print(f"操作タイプ: {changes['operation_type']}")
                if changes['operation_type'] == 'modified':
                    print(f"CCN変化量: {changes['ccn_change']:+d}")
                    print(f"長さ変化量: {changes['length_change']:+d}")
                    print(f"トークン変化量: {changes['tokens_change']:+d}")
                elif changes['operation_type'] == 'added':
                    print(f"メソッドが新たに追加されました")
                elif changes['operation_type'] == 'deleted':
                    print(f"メソッドが削除されました")
                elif changes['operation_type'] == 'NaN':
                    print(f"メソッドの分析に失敗しました")

                # 結果を保存
                complexity_results[record_id] = changes
                processed_count += 1

            except Exception as e:
                print(f"エラー: レコード {record_id} の処理中に例外が発生しました: {e}")
                # 例外が発生した場合もNaNタイプとして記録
                complexity_results[record_id] = {
                    'current_commit': None,
                    'ccn_change': None,
                    'length_change': None,
                    'tokens_change': None,
                    'operation_type': 'NaN'
                }
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
        print(f"変化量データが '{enhanced_output_csv}' に保存されました")

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
            # 各レコードごとの変化量の統計
            if complexity_results:
                # 成功したレコードの変化量データを取得
                successful_changes = [changes for changes in complexity_results.values() if changes is not None]

                # operation_typeごとの集計
                operation_counts = {}
                for changes in successful_changes:
                    op_type = changes['operation_type']
                    operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

                print(f"操作タイプ別集計:")
                for op_type, count in operation_counts.items():
                    percentage = count / len(successful_changes) * 100
                    print(f"  {op_type}: {count}個 ({percentage:.1f}%)")

                # modifiedのみの変化量統計
                modified_only = [changes for changes in successful_changes if changes['operation_type'] == 'modified']

                if modified_only:
                    print(f"\n=== 変更されたメソッドの統計 (n={len(modified_only)}) ===")

                    ccn_changes = [changes['ccn_change'] for changes in modified_only]
                    length_changes = [changes['length_change'] for changes in modified_only]
                    tokens_changes = [changes['tokens_change'] for changes in modified_only]

                    import numpy as np

                    print(f"\n=== 変更されたメソッドの変化量統計 ===")

                    if ccn_changes:
                        print(f"CCN変化量:")
                        print(f"  平均: {np.mean(ccn_changes):.3f}")
                        print(f"  中央値: {np.median(ccn_changes):.3f}")
                        print(f"  標準偏差: {np.std(ccn_changes):.3f}")
                        print(f"  最小値: {np.min(ccn_changes)}")
                        print(f"  最大値: {np.max(ccn_changes)}")

                    if length_changes:
                        print(f"\n長さ変化量:")
                        print(f"  平均: {np.mean(length_changes):.3f}")
                        print(f"  中央値: {np.median(length_changes):.3f}")
                        print(f"  標準偏差: {np.std(length_changes):.3f}")
                        print(f"  最小値: {np.min(length_changes)}")
                        print(f"  最大値: {np.max(length_changes)}")

                    if tokens_changes:
                        print(f"\nトークン変化量:")
                        print(f"  平均: {np.mean(tokens_changes):.3f}")
                        print(f"  中央値: {np.median(tokens_changes):.3f}")
                        print(f"  標準偏差: {np.std(tokens_changes):.3f}")
                        print(f"  最小値: {np.min(tokens_changes)}")
                        print(f"  最大値: {np.max(tokens_changes)}")

                    # 変化のタイプ別集計（modifiedのみ）
                    ccn_increases = sum(1 for change in ccn_changes if change > 0)
                    ccn_decreases = sum(1 for change in ccn_changes if change < 0)
                    ccn_no_change = sum(1 for change in ccn_changes if change == 0)

                    length_increases = sum(1 for change in length_changes if change > 0)
                    length_decreases = sum(1 for change in length_changes if change < 0)
                    length_no_change = sum(1 for change in length_changes if change == 0)

                    tokens_increases = sum(1 for change in tokens_changes if change > 0)
                    tokens_decreases = sum(1 for change in tokens_changes if change < 0)
                    tokens_no_change = sum(1 for change in tokens_changes if change == 0)

                    print(f"\n=== 変更されたメソッドの変化の方向性 ===")
                    print(f"CCN増加: {ccn_increases}個 ({ccn_increases/len(ccn_changes)*100:.1f}%)")
                    print(f"CCN減少: {ccn_decreases}個 ({ccn_decreases/len(ccn_changes)*100:.1f}%)")
                    print(f"CCN変化なし: {ccn_no_change}個 ({ccn_no_change/len(ccn_changes)*100:.1f}%)")

                    print(f"\n長さ増加: {length_increases}個 ({length_increases/len(length_changes)*100:.1f}%)")
                    print(f"長さ減少: {length_decreases}個 ({length_decreases/len(length_changes)*100:.1f}%)")
                    print(f"長さ変化なし: {length_no_change}個 ({length_no_change/len(length_changes)*100:.1f}%)")

                    print(f"\nトークン増加: {tokens_increases}個 ({tokens_increases/len(tokens_changes)*100:.1f}%)")
                    print(f"トークン減少: {tokens_decreases}個 ({tokens_decreases/len(tokens_changes)*100:.1f}%)")
                    print(f"トークン変化なし: {tokens_no_change}個 ({tokens_no_change/len(tokens_changes)*100:.1f}%)")
                else:
                    print("\n変更されたメソッドがありませんでした")

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
