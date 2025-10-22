import pandas as pd
import numpy as np

def drop_zero_columns(input_path, output_path):
    try:
        df = pd.read_csv(input_path)

        print(f"元のデータ: {df.shape[0]}行 × {df.shape[1]}列")

        zero_columns = []
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                if (df[column] == 0).all():
                    zero_columns.append(column)

        print(f"削除対象のカラム数: {len(zero_columns)}")
        if zero_columns:
            print("削除対象のカラム:")
            for col in zero_columns:
                print(f"  - {col}")

        df_filtered = df.drop(columns=zero_columns)

        print(f"処理後のデータ: {df_filtered.shape[0]}行 × {df_filtered.shape[1]}列")

        df_filtered.to_csv(output_path, index=False)
        print(f"結果を {output_path} に保存しました")

        return df_filtered

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_path}' が見つかりません")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def main():
    base_dir = "../data/remove/neo4j/"
    input_path = base_dir + "method-p.csv"
    output_path = base_dir + "method-p_drop_columns.csv"

    result = drop_zero_columns(input_path, output_path)

    if result is not None:
        print("\n処理が完了しました！")
        print(f"残ったカラム数: {result.shape[1]}")
        print("最初の5行のプレビュー:")
        print(result.head())

if __name__ == "__main__":
    main()
