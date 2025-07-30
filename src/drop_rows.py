import pandas as pd

def filter_csv_file(input_path, output_path):
    try:
        print(f"CSVファイルを読み込んでいます: {input_path}")
        df = pd.read_csv(input_path)

        original_count = len(df)
        print(f"元のレコード数: {original_count}")

        if 'LongName' not in df.columns:
            raise ValueError("LongNameカラムが見つかりません")

        lucene_mask = df['LongName'].str.startswith('org.apache.lucene', na=False)
        lucene_count = lucene_mask.sum()
        print(f"org.apache.luceneで始まるレコード数: {lucene_count}")

        if lucene_count > 0:
            print("\n削除されるレコードのサンプル:")
            sample_records = df[lucene_mask]['LongName'].head(5)
            for i, long_name in enumerate(sample_records, 1):
                print(f"  {i}. {long_name}")
            if lucene_count > 5:
                print(f"  ... その他 {lucene_count - 5} 件")

        filtered_df = df[~lucene_mask]

        filtered_count = len(filtered_df)
        print(f"\nフィルタリング後のレコード数: {filtered_count}")
        print(f"削除されたレコード数: {original_count - filtered_count}")

        print(f"\n結果を保存しています: {output_path}")
        filtered_df.to_csv(output_path, index=False)

        print("フィルタリングが完了しました！")

        print(f"\n=== 統計情報 ===")
        print(f"元のレコード数: {original_count:,}")
        print(f"削除されたレコード数: {lucene_count:,}")
        print(f"残ったレコード数: {filtered_count:,}")
        print(f"削除率: {(lucene_count / original_count * 100):.2f}%")

        return True

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_path}' が見つかりません")
        return False
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

def main():
    base_dir = "../data/remove/ceylon-ide-eclipse/"
    input_path = base_dir + "method-p_drop_columns.csv"
    output_path = base_dir + "method-p_drop_columns_rows.csv"

    print("CSVフィルタリングプログラム（pandas使用）")
    print("=" * 50)
    print("LongNameカラムの値が'org.apache.lucene'で始まるレコードを削除します。")
    print()

    success = filter_csv_file(input_path, output_path)

    if success:
        print(f"\n✓ 処理が正常に完了しました。")
        print(f"出力ファイル: {output_path}")
    else:
        print("\n✗ 処理中にエラーが発生しました。")

if __name__ == "__main__":
    main()
