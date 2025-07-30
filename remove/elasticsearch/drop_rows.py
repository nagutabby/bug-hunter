import pandas as pd

def filter_csv_file(input_file, output_file):
    """
    CSVファイルからorg.apache.luceneで始まるLongNameのレコードを除外する

    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力CSVファイルのパス
    """
    try:
        # CSVファイルを読み込み
        print(f"CSVファイルを読み込んでいます: {input_file}")
        df = pd.read_csv(input_file)

        # 元のレコード数を記録
        original_count = len(df)
        print(f"元のレコード数: {original_count}")

        # LongNameカラムが存在することを確認
        if 'LongName' not in df.columns:
            raise ValueError("LongNameカラムが見つかりません")

        # org.apache.luceneで始まるレコードの数を確認
        lucene_mask = df['LongName'].str.startswith('org.apache.lucene', na=False)
        lucene_count = lucene_mask.sum()
        print(f"org.apache.luceneで始まるレコード数: {lucene_count}")

        # サンプルを表示
        if lucene_count > 0:
            print("\n削除されるレコードのサンプル:")
            sample_records = df[lucene_mask]['LongName'].head(5)
            for i, long_name in enumerate(sample_records, 1):
                print(f"  {i}. {long_name}")
            if lucene_count > 5:
                print(f"  ... その他 {lucene_count - 5} 件")

        # org.apache.luceneで始まらないレコードのみを保持
        filtered_df = df[~lucene_mask]

        # フィルタリング後のレコード数
        filtered_count = len(filtered_df)
        print(f"\nフィルタリング後のレコード数: {filtered_count}")
        print(f"削除されたレコード数: {original_count - filtered_count}")

        # 結果をCSVファイルに保存
        print(f"\n結果を保存しています: {output_file}")
        filtered_df.to_csv(output_file, index=False)

        print("フィルタリングが完了しました！")

        # 統計情報を表示
        print(f"\n=== 統計情報 ===")
        print(f"元のレコード数: {original_count:,}")
        print(f"削除されたレコード数: {lucene_count:,}")
        print(f"残ったレコード数: {filtered_count:,}")
        print(f"削除率: {(lucene_count / original_count * 100):.2f}%")

        return True

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_file}' が見つかりません")
        return False
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

def main():
    # ファイルパスの設定
    input_file = "method-p_drop_columns.csv"
    output_file = "method-p_drop_columns_rows.csv"

    print("CSVフィルタリングプログラム（pandas使用）")
    print("=" * 50)
    print("LongNameカラムの値が'org.apache.lucene'で始まるレコードを削除します。")
    print()

    # フィルタリングを実行
    success = filter_csv_file(input_file, output_file)

    if success:
        print(f"\n✓ 処理が正常に完了しました。")
        print(f"出力ファイル: {output_file}")
    else:
        print("\n✗ 処理中にエラーが発生しました。")

if __name__ == "__main__":
    main()
