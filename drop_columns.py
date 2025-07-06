import pandas as pd
import numpy as np

def drop_zero_columns(input_file, output_file):
    """
    CSVファイルから全ての値がゼロであるカラムを削除し、新しいCSVファイルを作成する

    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力CSVファイルのパス
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(input_file)

        print(f"元のデータ: {df.shape[0]}行 × {df.shape[1]}列")

        # 全ての値がゼロであるカラムを特定
        zero_columns = []
        for column in df.columns:
            # 数値型の列のみチェック（文字列列は除外）
            if pd.api.types.is_numeric_dtype(df[column]):
                # 全ての値がゼロかチェック
                if (df[column] == 0).all():
                    zero_columns.append(column)

        print(f"削除対象のカラム数: {len(zero_columns)}")
        if zero_columns:
            print("削除対象のカラム:")
            for col in zero_columns:
                print(f"  - {col}")

        # 全ゼロカラムを削除
        df_filtered = df.drop(columns=zero_columns)

        print(f"処理後のデータ: {df_filtered.shape[0]}行 × {df_filtered.shape[1]}列")

        # 新しいCSVファイルに保存
        df_filtered.to_csv(output_file, index=False)
        print(f"結果を {output_file} に保存しました")

        return df_filtered

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_file}' が見つかりません")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def main():
    # ファイルパスを指定
    input_file = "method-p.csv"
    output_file = "method-p_filtered.csv"

    # 処理実行
    result = drop_zero_columns(input_file, output_file)

    if result is not None:
        print("\n処理が完了しました！")
        print(f"残ったカラム数: {result.shape[1]}")
        print("最初の5行のプレビュー:")
        print(result.head())

if __name__ == "__main__":
    main()
