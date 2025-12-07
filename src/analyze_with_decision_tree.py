import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
import os
from typing import Optional
import warnings
from train import BugHunterTrainer, JavaCodeTokenizer

warnings.filterwarnings('ignore')


class DecisionTreeAnalyzer:
    def __init__(self, model_path: str, random_state: int = 42):
        self.model_path = model_path
        self.random_state = random_state
        self.bug_hunter_trainer = None
        self.original_data = None
        self.X_processed = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.simple_tree = None

        self._load_pretrained_model()

    def _load_pretrained_model(self):
        print(f"モデルファイルの読み込み: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイル '{self.model_path}' が見つかりません")

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.bug_hunter_trainer = BugHunterTrainer()

            required_keys = ['tfidf_vectorizer_longname', 'tfidf_vectorizer_parent', 'scaler',
                           'operation_type_columns', 'all_feature_names', 'has_operation_type', 'java_tokenizer']

            for key in required_keys:
                if key in model_data:
                    setattr(self.bug_hunter_trainer, key, model_data[key])
                else:
                    print(f"  警告: {key} がモデルデータに見つかりません")

            print(f"事前訓練されたBugHunterモデルを読み込みました")

        except Exception as e:
            import traceback
            print(f"_load_pretrained_modelでエラーが発生: {e}")
            raise

    def load_and_prepare_data(self, data_path: str, max_rows: int = None):
        print(f"\n=== データ読み込みと前処理 ===")
        print(f"データパス: {data_path}")

        try:
            if max_rows:
                self.original_data = pd.read_csv(data_path, nrows=max_rows)
            else:
                self.original_data = pd.read_csv(data_path)

            if "Number of Bugs" not in self.original_data.columns:
                raise ValueError("ターゲットカラム 'Number of Bugs' がデータに見つかりません。")

            self.y = (self.original_data["Number of Bugs"] > 0.5).astype(int)

            result = self.bug_hunter_trainer.prepare_data(
                self.original_data, is_training=False
            )

            if isinstance(result, tuple) and len(result) == 2:
                self.X_processed, _ = result
            else:
                raise ValueError(f"prepare_dataが期待される形式を返しませんでした")

            self.X_processed = self.X_processed.loc[self.y.index]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_processed, self.y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.y
            )

            print(f"訓練データ: {len(self.X_train)}件, テストデータ: {len(self.X_test)}件")

        except Exception as e:
            import traceback
            print(f"load_and_prepare_dataでエラーが発生: {e}")
            raise

    def build_simple_decision_tree(self, max_depth: int = 5, min_samples_split: int = 100,
                                 min_samples_leaf: int = 50):
        print(f"\n=== 簡易決定木の構築 ===")

        self.simple_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'
        )

        self.simple_tree.fit(self.X_train, self.y_train)

        y_pred = self.simple_tree.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"テスト精度: {accuracy:.4f}")

    def build_simple_decision_tree_for_display(self, max_depth: int = 3, min_samples_split: int = 100,
                                min_samples_leaf: int = 50):
        """表示用に深さを制限した決定木を別途構築"""
        print(f"\n=== 表示用決定木の構築 ===")

        display_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'
        )

        display_tree.fit(self.X_train, self.y_train)
        return display_tree

    def plot_decision_tree_clean(self, save_path: Optional[str] = None, max_depth_display: int = 3):
        print(f"\n=== 決定木可視化（クリーン版） ===")

        display_tree = self.build_simple_decision_tree_for_display(max_depth=max_depth_display)

        feature_names = self.X_processed.columns.tolist()
        short_feature_names = [self._shorten_feature_name(name) for name in feature_names]

        def get_node_color(node_id, tree_obj):
            if node_id >= len(tree_obj.value):
                return "lightgray"

            value = tree_obj.value[node_id]
            impurity = tree_obj.impurity[node_id]

            negative_samples = value[0][0]
            positive_samples = value[0][1]

            majority_class = 1 if positive_samples > negative_samples else 0

            if majority_class == 0:  # 陰性クラス（バグなし）→ 青系
                if impurity < 0.1: color = "dodgerblue"
                elif impurity < 0.3: color = "lightblue"
                else: color = "lightcyan"
            else:  # 陽性クラス（バグあり）→ オレンジ系
                if impurity < 0.1: color = "darkorange"
                elif impurity < 0.3: color = "orange"
                else: color = "moccasin"
            return color

        fig, ax = plt.subplots(figsize=(25, 15))

        artists = plot_tree(
            display_tree,
            feature_names=short_feature_names,
            class_names=['陰性', '陽性'],
            filled=True,
            rounded=True,
            fontsize=14,
            ax=ax
        )

        tree_obj = display_tree.tree_
        node_count = 0

        for artist in artists:
            if hasattr(artist, 'get_bbox_patch') and artist.get_bbox_patch() is not None:
                if node_count < len(tree_obj.value):
                    color = get_node_color(node_count, tree_obj)
                    artist.get_bbox_patch().set_facecolor(color)
                    artist.get_bbox_patch().set_edgecolor('black')
                    artist.get_bbox_patch().set_linewidth(1.5)
                    node_count += 1

        plt.tight_layout()

        if save_path is None:
            save_path = "decision_tree_visualization.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"決定木を '{save_path}' に保存しました")

    def print_detailed_tree_stats(self, max_depth_limit: int = 3):
        """
        決定木の詳細な統計情報、ノード情報、ルール抽出を行い標準出力に表示する
        (全ての出力を max_depth_limit の深さまでに制限する)
        ※項目1（ノード詳細）と項目5（比較情報）は削除
        """
        if self.simple_tree is None:
            print("エラー: 決定木が構築されていません。")
            return

        tree = self.simple_tree.tree_
        feature_names = self.X_processed.columns
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        values = tree.value
        impurity = tree.impurity

        # 統計用変数
        n_displayed_nodes = 0
        n_displayed_leaves = 0
        n_pos_leaves = 0
        n_neg_leaves = 0
        used_features = set()

        pos_leaves_details = []
        neg_leaves_details = []

        # === 1. 各ノードの詳細情報の計算（出力は省略） ===

        stack = [(0, 0)]

        while stack:
            node_id, depth = stack.pop()

            if depth > max_depth_limit:
                continue

            n_displayed_nodes += 1

            is_real_leaf = (children_left[node_id] == children_right[node_id])
            is_report_leaf = is_real_leaf or (depth == max_depth_limit)

            weighted_n_samples = np.sum(values[node_id])

            if weighted_n_samples < 1e-9:
                pos_ratio = 0.0
                neg_ratio = 0.0
                majority = "不明"
            else:
                val = values[node_id][0]
                neg_count = val[0]
                pos_count = val[1]
                pos_ratio = pos_count / weighted_n_samples
                neg_ratio = neg_count / weighted_n_samples
                majority = "陽性" if pos_count >= neg_count else "陰性"

            # 統計情報の収集
            if not is_report_leaf:
                fname = feature_names[feature[node_id]]
                short_fname = self._shorten_feature_name(fname)
                used_features.add(short_fname)

            # リーフ情報の収集
            if is_report_leaf:
                n_displayed_leaves += 1
                if majority == "陽性":
                    n_pos_leaves += 1
                    pos_leaves_details.append((node_id, pos_ratio, weighted_n_samples))
                else:
                    n_neg_leaves += 1
                    neg_leaves_details.append((node_id, neg_ratio, weighted_n_samples))
            else:
                stack.append((children_right[node_id], depth + 1))
                stack.append((children_left[node_id], depth + 1))

        # === 以下、レポート出力 ===

        print("\n" + "="*80)
        print(f"1. リーフノードの分類別集計 (深さ{max_depth_limit}時点での集計)")
        print("="*80)

        pos_leaves_details.sort(key=lambda x: x[1], reverse=True)
        neg_leaves_details.sort(key=lambda x: x[1], reverse=True)

        print(f"陽性クラスに分類するリーフノード (計{len(pos_leaves_details)}個):")
        print(f"  [番号リスト]: {', '.join([str(x[0]) for x in pos_leaves_details])}")
        for nid, ratio, samples in pos_leaves_details:
            print(f"  - ノード{nid}: 陽性割合={ratio:.3f}, サンプル数(重み付)={samples:.1f}")

        print(f"\n陰性クラスに分類するリーフノード (計{len(neg_leaves_details)}個):")
        print(f"  [番号リスト]: {', '.join([str(x[0]) for x in neg_leaves_details])}")
        for nid, ratio, samples in neg_leaves_details:
            print(f"  - ノード{nid}: 陰性割合={ratio:.3f}, サンプル数(重み付)={samples:.1f}")

        print("\n" + "="*80)
        print("2. 木構造の統計情報 (表示範囲内)")
        print("="*80)
        print(f"- 表示ノード数: {n_displayed_nodes}")
        print(f"- 表示リーフ数: {n_displayed_leaves}")
        print(f"- 陽性リーフ数: {n_pos_leaves}")
        print(f"- 陰性リーフ数: {n_neg_leaves}")
        print(f"- 最大深さ: {max_depth_limit} (制限)")
        print(f"- 使用された特徴量 ({len(used_features)}個): {list(used_features)}")

        print("\n" + "="*80)
        print(f"3. 代表的な分類ルール (Rootからのパス, Max Depth={max_depth_limit})")
        print("="*80)

        def get_rules(node_id, current_path, depth):
            is_real_leaf = (children_left[node_id] == children_right[node_id])
            is_report_leaf = is_real_leaf or (depth == max_depth_limit)

            if is_report_leaf:
                weighted_n_samples = np.sum(values[node_id])
                if weighted_n_samples < 1e-9:
                    return

                val = values[node_id][0]
                pos_ratio = val[1] / weighted_n_samples
                neg_ratio = val[0] / weighted_n_samples

                if pos_ratio >= 0.7:
                    print(f"陽性に分類 (確信度 {pos_ratio:.3f}):")
                    print(f"  ノード{node_id}: {' AND '.join(current_path)}")
                    print(f"  → 陽性割合={pos_ratio:.3f}, サンプル数(重み付)={weighted_n_samples:.1f}")
                    print("-" * 40)

                if neg_ratio >= 0.8:
                    print(f"陰性に分類 (確信度 {neg_ratio:.3f}):")
                    print(f"  ノード{node_id}: {' AND '.join(current_path)}")
                    print(f"  → 陰性割合={neg_ratio:.3f}, サンプル数(重み付)={weighted_n_samples:.1f}")
                    print("-" * 40)
                return

            fname = feature_names[feature[node_id]]
            short_fname = self._shorten_feature_name(fname)
            th = threshold[node_id]

            get_rules(children_left[node_id], current_path + [f"({short_fname} <= {th:.3f})"], depth + 1)
            get_rules(children_right[node_id], current_path + [f"({short_fname} > {th:.3f})"], depth + 1)

        get_rules(0, [], 0)

        # 項目5は削除済み

    def _shorten_feature_name(self, feature_name: str, max_length: int = 25) -> str:
        if len(feature_name) <= max_length:
            return feature_name

        if feature_name.startswith('LongName_tfidf_'):
            return f"LN_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('Parent_tfidf_'):
            return f"P_tf_{feature_name.split('_')[-1]}"
        elif feature_name.startswith('operation_type_'):
            op_type = feature_name.replace('operation_type_', '')
            return f"op_{op_type[:10]}"
        else:
            if len(feature_name) > max_length:
                return feature_name[:max_length-3] + "..."
            return feature_name

    def generate_decision_tree_report(self, data_path: str, output_dir: str = ".", max_rows: int = None):
        print("="*80)
        print("決定木分析レポート生成開始")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        self.load_and_prepare_data(data_path, max_rows)

        self.build_simple_decision_tree(max_depth=5, min_samples_split=20, min_samples_leaf=10)

        self.print_detailed_tree_stats(max_depth_limit=3)

        tree_viz_path = os.path.join(output_dir, "decision_tree_visualization.png")
        self.plot_decision_tree_clean(save_path=tree_viz_path, max_depth_display=3)

        print("\n" + "="*80)
        print("処理完了")
        print("="*80)


def main():
    try:
        project_name = "orientdb"
        base_dir = f"../data/remove/{project_name}/"
        output_dir = f"../materials/images/{project_name}/"

        analyzer = DecisionTreeAnalyzer(
            model_path=base_dir + "predictions_add_method_commit_level_metrics.pkl"
        )

        data_path = base_dir + "method-p_add_method_commit_level_metrics.csv"

        analyzer.generate_decision_tree_report(
            data_path=data_path,
            output_dir=output_dir,
            max_rows=5000
        )

    except Exception as e:
        import traceback
        print(f"エラー: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
