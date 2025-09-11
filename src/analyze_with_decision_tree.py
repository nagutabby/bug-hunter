import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import pickle
import os
from typing import Dict, List, Tuple, Optional
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

            print(f"読み込まれたmodel_dataのキー: {list(model_data.keys())}")

            self.bug_hunter_trainer = BugHunterTrainer()

            required_keys = ['tfidf_vectorizer_longname', 'tfidf_vectorizer_parent', 'scaler',
                           'operation_type_columns', 'all_feature_names', 'has_operation_type', 'java_tokenizer']

            for key in required_keys:
                if key in model_data:
                    setattr(self.bug_hunter_trainer, key, model_data[key])
                    print(f"  {key}: 読み込み成功")
                else:
                    print(f"  警告: {key} がモデルデータに見つかりません")

            print(f"事前訓練されたBugHunterモデルを読み込みました")

        except Exception as e:
            import traceback
            print(f"_load_pretrained_modelでエラーが発生: {e}")
            print("詳細なトレースバック:")
            print(traceback.format_exc())
            raise

    def load_and_prepare_data(self, data_path: str, max_rows: int = None):
        print(f"\n=== データ読み込みと前処理 ===")
        print(f"データパス: {data_path}")

        try:
            if max_rows:
                self.original_data = pd.read_csv(data_path, nrows=max_rows)
                print(f"読み込み行数: {max_rows}行 (制限)")
            else:
                self.original_data = pd.read_csv(data_path)
                print(f"読み込み行数: {len(self.original_data)}行 (全データ)")

            print(f"カラム数: {len(self.original_data.columns)}")

            if "Number of Bugs" not in self.original_data.columns:
                raise ValueError("ターゲットカラム 'Number of Bugs' がデータに見つかりません。")

            self.y = (self.original_data["Number of Bugs"] > 0.5).astype(int)
            print(f"ターゲット変数yを準備しました。分布: 陰性={sum(self.y==0)}, 陽性={sum(self.y==1)}")

            print("\nBugHunterTrainer.prepare_dataを呼び出し中...")
            result = self.bug_hunter_trainer.prepare_data(
                self.original_data, is_training=False
            )

            if result is None:
                raise ValueError("prepare_dataがNoneを返しました")

            if isinstance(result, tuple) and len(result) == 2:
                self.X_processed, _ = result
                print(f"X_processed: {type(self.X_processed)}, shape: {self.X_processed.shape if hasattr(self.X_processed, 'shape') else 'No shape'}")
            else:
                raise ValueError(f"prepare_dataが期待される形式を返しませんでした: {result}")

            if self.X_processed is None:
                raise ValueError("X_processedがNoneです")

            print(f"前処理後の特徴量数: {self.X_processed.shape[1]}")

            self.X_processed = self.X_processed.loc[self.y.index]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_processed, self.y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.y
            )

            print(f"\n訓練データ: {len(self.X_train)}件")
            print(f"テストデータ: {len(self.X_test)}件")

        except Exception as e:
            import traceback
            print(f"load_and_prepare_dataでエラーが発生: {e}")
            print("詳細なトレースバック:")
            print(traceback.format_exc())
            raise

    def build_simple_decision_tree(self, max_depth: int = 5, min_samples_split: int = 100,
                                 min_samples_leaf: int = 50):
        print(f"\n=== 簡易決定木の構築 ===")
        print(f"パラメータ: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

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

        cv_scores = cross_val_score(
            self.simple_tree, self.X_train, self.y_train,
            cv=5, scoring='f1'
        )

        print(f"テスト精度: {accuracy:.4f}")
        print(f"5分割交差検証F1スコア: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print(f"\n=== 分類レポート ===")
        print(classification_report(self.y_test, y_pred,
                                  target_names=['陰性(バグなし)', '陽性(バグあり)']))

    def analyze_tree_structure(self, top_n_features: int = 10):
        print(f"\n=== 決定木構造分析 ===")

        feature_names = self.X_processed.columns.tolist()
        feature_importances = self.simple_tree.feature_importances_

        importance_df = pd.DataFrame({
            '特徴量': feature_names,
            '重要度': feature_importances
        }).sort_values('重要度', ascending=False)

        print(f"上位{top_n_features}重要特徴量:")
        for i, (_, row) in enumerate(importance_df.head(top_n_features).iterrows()):
            feature_type = self._get_feature_type(row['特徴量'])
            print(f"  {i+1:2d}. {row['特徴量']} ({feature_type}): {row['重要度']:.4f}")

        print(f"\n=== 決定木の分岐ルール（テキスト形式） ===")
        tree_rules = export_text(
            self.simple_tree,
            feature_names=feature_names,
            max_depth=3
        )
        print(tree_rules)

        return importance_df

    def plot_decision_tree(self, save_path: Optional[str] = None, max_depth_display: int = 3):
        print(f"\n=== 決定木可視化 ===")

        feature_names = self.X_processed.columns.tolist()
        short_feature_names = [self._shorten_feature_name(name) for name in feature_names]

        plt.figure(figsize=(20, 12))
        plot_tree(
            self.simple_tree,
            feature_names=short_feature_names,
            class_names=['陰性', '陽性'],
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth_display,
            impurity=False,  # 不純度を非表示
            proportion=True  # サンプル数を割合で表示
        )

        plt.title(f'決定木 (深さ{max_depth_display})', fontsize=16, pad=20)
        plt.tight_layout()

        if save_path is None:
            save_path = "decision_tree_visualization.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"決定木を '{save_path}' に保存しました")
        plt.show()

    def build_simple_decision_tree_for_display(self, max_depth: int = 3, min_samples_split: int = 100,
                                min_samples_leaf: int = 50):
        """表示用に深さを制限した決定木を別途構築"""
        print(f"\n=== 表示用決定木の構築 ===")
        print(f"パラメータ: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

        display_tree = DecisionTreeClassifier(
            max_depth=max_depth,  # 表示用は深さ3に制限
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'
        )

        display_tree.fit(self.X_train, self.y_train)
        return display_tree

    def plot_decision_tree_clean(self, save_path: Optional[str] = None, max_depth_display: int = 3):
        """(...）ノードを完全に除去した決定木表示"""
        print(f"\n=== 決定木可視化（クリーン版） ===")

        # 表示用に深さ制限した決定木を作成
        display_tree = self.build_simple_decision_tree_for_display(max_depth=max_depth_display)

        feature_names = self.X_processed.columns.tolist()
        short_feature_names = [self._shorten_feature_name(name) for name in feature_names]

        plt.figure(figsize=(20, 12))
        plot_tree(
            display_tree,  # 深さ制限された決定木を使用
            feature_names=short_feature_names,
            class_names=['陰性', '陽性'],
            filled=True,
            rounded=True,
            fontsize=10
        )

        plt.title(f'決定木 (深さ{max_depth_display})', fontsize=16, pad=20)
        plt.tight_layout()

        if save_path is None:
            save_path = "decision_tree_visualization.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"決定木を '{save_path}' に保存しました")
        plt.show()

    def plot_feature_importance(self, save_path: Optional[str] = None, top_n: int = 10):
        print(f"\n=== 特徴量重要度可視化 ===")

        feature_names = self.X_processed.columns.tolist()
        feature_importances = self.simple_tree.feature_importances_

        importance_df = pd.DataFrame({
            '特徴量': feature_names,
            '重要度': feature_importances,
            'タイプ': [self._get_feature_type(f) for f in feature_names]
        }).sort_values('重要度', ascending=False)

        top_features = importance_df.head(top_n)

        type_colors = {
            'LongName TF-IDF': '#FF6B6B',
            'Parent TF-IDF': '#4ECDC4',
            'operation_type': '#45B7D1',
            '数値': '#96CEB4'
        }

        colors = [type_colors.get(t, '#95A5A6') for t in top_features['タイプ']]
        short_names = [self._shorten_feature_name(f, 25) for f in top_features['特徴量']]

        plt.figure(figsize=(12, max(8, len(top_features) * 0.4)))

        y_pos = np.arange(len(top_features))
        bars = plt.barh(y_pos, top_features['重要度'], color=colors, alpha=0.8, edgecolor='black')

        plt.yticks(y_pos, short_names)
        plt.xlabel('特徴量重要度', fontsize=12)
        plt.title(f'決定木 特徴量重要度 Top {len(top_features)}', fontsize=14, pad=20)

        for i, (bar, importance) in enumerate(zip(bars, top_features['重要度'])):
            width = bar.get_width()
            plt.text(width + max(top_features['重要度']) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontsize=9)

        unique_types = list(set(top_features['タイプ']))
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=type_colors.get(t, '#95A5A6'),
                                      alpha=0.8, edgecolor='black') for t in unique_types]
        plt.legend(legend_elements, unique_types, loc='lower right', fontsize=10)

        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, linestyle='--', axis='x')
        plt.tight_layout()

        if save_path is None:
            save_path = "decision_tree_feature_importance.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特徴量重要度チャートを '{save_path}' に保存しました")
        plt.show()

        return importance_df

    def analyze_decision_paths(self, max_depth_analysis: int = 5):
        print(f"\n=== 決定パス分析 ===")

        feature_names = self.X_processed.columns.tolist()

        tree = self.simple_tree.tree_

        def get_leaf_paths(node_id=0, path=[], depth=0):
            if depth > max_depth_analysis:
                return []

            if tree.children_left[node_id] == tree.children_right[node_id]:
                return [path.copy()]

            paths = []

            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature]

            left_condition = f"{self._shorten_feature_name(feature_name, 20)} <= {threshold:.3f}"
            right_condition = f"{self._shorten_feature_name(feature_name, 20)} > {threshold:.3f}"

            path.append(left_condition)
            paths.extend(get_leaf_paths(tree.children_left[node_id], path, depth + 1))
            path.pop()

            path.append(right_condition)
            paths.extend(get_leaf_paths(tree.children_right[node_id], path, depth + 1))
            path.pop()

            return paths

        all_paths = get_leaf_paths()

        print(f"決定木の主要な分岐パス（上位10パス）:")
        for i, path in enumerate(all_paths[:10]):
            print(f"  パス {i+1}: {' → '.join(path)}")

        feature_usage = {}
        for path in all_paths:
            for condition in path:
                feature = condition.split(' ')[0]
                feature_usage[feature] = feature_usage.get(feature, 0) + 1

        sorted_usage = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        print(f"\n分岐で最も使用される特徴量:")
        for i, (feature, count) in enumerate(sorted_usage[:10]):
            print(f"  {i+1:2d}. {feature}: {count}回使用")

        return all_paths, feature_usage

    def _get_feature_type(self, feature_name: str) -> str:
        if feature_name.startswith('LongName_tfidf_'):
            return 'LongName TF-IDF'
        elif feature_name.startswith('Parent_tfidf_'):
            return 'Parent TF-IDF'
        elif feature_name.startswith('operation_type_'):
            return 'operation_type'
        else:
            return '数値'

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
        print("決定木分析レポート")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        self.load_and_prepare_data(data_path, max_rows)

        # 分析用の深い決定木を構築
        self.build_simple_decision_tree(max_depth=5, min_samples_split=100, min_samples_leaf=50)

        # 表示用のクリーンな決定木を使用
        tree_viz_path = os.path.join(output_dir, "decision_tree_visualization.png")
        self.plot_decision_tree_clean(save_path=tree_viz_path, max_depth_display=3)

        importance_path = os.path.join(output_dir, "decision_tree_feature_importance.png")
        importance_df = self.plot_feature_importance(save_path=importance_path, top_n=10)

        structure_df = self.analyze_tree_structure(top_n_features=10)

        paths, feature_usage = self.analyze_decision_paths(max_depth_analysis=5)

        print("\n" + "="*80)
        print("決定木分析完了")
        print(f"出力ディレクトリ: {output_dir}")
        print("生成されたファイル:")
        print("  - decision_tree_visualization.png (決定木可視化)")
        print("  - decision_tree_feature_importance.png (特徴量重要度)")
        print("="*80)

        return {
            'tree_importance': importance_df,
            'structure_analysis': structure_df,
            'decision_paths': paths,
            'feature_usage': feature_usage
        }


def main():
    try:
        project_name = "oryx"
        base_dir = f"../data/remove/{project_name}/"
        output_dir = f"../materials/images/{project_name}/"

        analyzer = DecisionTreeAnalyzer(
            model_path=base_dir + "predictions_add_change_metrics.pkl"
        )

        data_path = base_dir + "method-p_add_change_metrics.csv"

        results = analyzer.generate_decision_tree_report(
            data_path=data_path,
            output_dir=output_dir,
            max_rows=5000
        )

        print("\n" + "="*60)
        print("決定木分析が完了しました！")
        print("="*60)

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("まず train.py を実行してモデルを訓練してください。")
    except Exception as e:
        import traceback
        print(f"エラーが発生しました: {e}")
        print("詳細なエラー情報:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
