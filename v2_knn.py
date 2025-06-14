from utils import BaseBugHunter, GLOBAL_SEED
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Categorical
import pandas as pd


class KNNBugHunter(BaseBugHunter):
    """
    k-NN版バグ予測クラス

    BaseBugHunterから継承し、k-NN固有の機能を実装
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 1000,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False):
        """
        コンストラクタ
        """
        super().__init__(feature_selection_threshold, tfidf_max_features,
                        java_tokenizer_min_length, include_package_tokens)

        # k-NN用の特徴量重要度計算用の補助モデル
        self.importance_model = None

    def evaluate_model_with_cv(self, params: dict, X: pd.DataFrame, y: pd.Series,
                              k_folds: int = 3) -> float:
        """交差検証を用いたk-NNモデル評価"""
        try:
            # k-NNモデルの作成
            knn = KNeighborsClassifier(
                n_neighbors=int(params['n_neighbors']),
                weights=params['weights'],
                metric=params['metric'],
                n_jobs=-1
            )

            # 交差検証でLog Lossを評価
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
            total_loss = 0.0

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                knn.fit(X_train_fold, y_train_fold)

                # 予測確率取得
                y_pred_proba = knn.predict_proba(X_val_fold)[:, 1]

                # Log Loss計算
                fold_loss = self.log_loss_function(y_val_fold, y_pred_proba)
                total_loss += fold_loss

            avg_loss = total_loss / k_folds
            return avg_loss

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')

    def optimize_hyperparameters_with_log_loss(self, X: pd.DataFrame, y: pd.Series,
                                                max_iterations: int = 15) -> dict:
        """Log Loss損失関数を用いたk-NN用ベイジアン最適化"""
        print("\n=== k-NN Log Lossベース ベイジアン最適化（カスタムJavaトークナイザー使用）===")
        print("最適化手法: Bayesian Optimization (scikit-optimize)")
        print("探索パラメータ: n_neighbors, weights, metric")
        print("クラス不均衡対応: ダウンサンプリング (事前に適用済み)")
        print("特徴量: 数値 + Java TF-IDF + One-Hot Encoding + 正規化")

        self.best_loss = float('inf')
        self.best_params = None
        self.optimization_history = []

        # k-NN用の探索空間の定義
        search_space = [
            Integer(10, 100, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Categorical(['euclidean', 'manhattan', 'cosine'], name='metric')
        ]

        # 目的関数
        def objective(params):
            n_neighbors, weights, metric = params

            param_dict = {
                'n_neighbors': int(n_neighbors),
                'weights': weights,
                'metric': metric,
            }

            # モデル評価
            loss = self.evaluate_model_with_cv(param_dict, X, y)

            # 履歴記録
            self.optimization_history.append({
                'params': param_dict.copy(),
                'loss': loss
            })

            # 最良パラメータ更新
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = param_dict.copy()
                print(f"新しい最良損失: {loss:.4f}")
                print(f"パラメータ: {param_dict}")

            return loss

        # Bayesian Optimization実行
        print("k-NN Bayesian Optimization開始...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=max_iterations,
            random_state=GLOBAL_SEED,
            acq_func='EI',
            n_initial_points=5
        )

        print(f"\nk-NN Bayesian Optimization完了!")
        print(f"最良損失: {self.best_loss:.4f}")
        final_best_params = self.best_params.copy() if self.best_params else {}
        print(f"最良パラメータ: {final_best_params}")
        print(f"総評価回数: {len(self.optimization_history)}")

        return final_best_params

    def train_initial_model_for_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """特徴量重要度を取得するための初期モデルを学習（k-NNは特徴量重要度を提供しないため、RandomForestで代替）"""
        print("\n=== 特徴量重要度取得のための補助モデル学習（RandomForest使用）===")

        # k-NNは特徴量重要度を提供しないため、RandomForestを補助的に使用
        self.importance_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        self.importance_model.fit(X, y)
        self.feature_importance = self.importance_model.feature_importances_
        print("補助モデル（RandomForest）学習と特徴量重要度の計算が完了しました。")
        return self.importance_model

    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, optimal_params: dict):
        """最適化されたパラメータでk-NNモデルを学習"""
        print("\n=== 最適化k-NNモデル学習 ===")

        knn_params = {
            'n_neighbors': int(optimal_params['n_neighbors']),
            'weights': optimal_params['weights'],
            'metric': optimal_params['metric'],
            'n_jobs': -1
        }

        self.best_model = KNeighborsClassifier(**knn_params)
        self.best_model.fit(X, y)

        print("最適化k-NNモデル学習完了")
        print(f"最終パラメータ: {knn_params}")
        print(f"学習データ: {len(X)}件")

        return self.best_model

    def run_pipeline(self, data_path: str):
        """k-NN用パイプラインの実行"""
        print("=== k-NN版ダウンサンプリングバグ予測パイプライン ===")

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. ダウンサンプリング適用
        X_downsampled, y_downsampled = self.apply_downsampling(X_full, y_full)
        print(f"ダウンサンプリング後データセットサイズ: {len(X_downsampled)}行")

        # 4. データ分割
        X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(
            X_downsampled, y_downsampled, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_downsampled
        )
        print(f"訓練データ: {len(X_train_ds)}行, テストデータ: {len(X_test_ds)}行")

        # 5. 特徴量重要度取得のための初期モデル学習（補助的にRandomForestを使用）
        self.train_initial_model_for_feature_importance(X_train_ds, y_train_ds)

        # 6. 特徴量削減
        X_train_reduced = self.select_features_by_importance(X_train_ds)
        X_test_reduced = self.select_features_by_importance(X_test_ds)

        # 7. k-NNハイパーパラメータ最適化
        optimal_params = self.optimize_hyperparameters_with_log_loss(
            X_train_reduced, y_train_ds, max_iterations=20
        )

        # 8. 最適化k-NNモデル学習
        optimized_model = self.train_optimized_model(
            X_train_reduced, y_train_ds, optimal_params
        )

        # 9. 評価
        results, y_pred, y_pred_proba = self.comprehensive_evaluation(
            X_test_reduced, y_test_ds
        )

        return results, optimal_params

    def display_feature_importance_table(self, top_n: int = 10):
        """特徴量重要度テーブルの表示（補助モデルのRandomForestから取得）"""
        if self.feature_importance is None or self.selected_features is None:
            print("特徴量重要度は計算されていません。または利用可能なモデルから取得できませんでした。")
            return

        feature_names = self.selected_features
        if len(feature_names) != len(self.feature_importance):
            if self.all_feature_names is not None and len(self.feature_importance) == len(self.all_feature_names):
                feature_names = self.all_feature_names
            else:
                print("エラー: 選択された特徴量名と補助モデルの重要度の数が一致しません。")
                return

        importance_df = pd.DataFrame({
            '特徴量': feature_names,
            '重要度': self.feature_importance
        }).sort_values('重要度', ascending=False)

        # self.selected_features に含まれる特徴量のみをフィルタリング
        importance_df_filtered = importance_df[importance_df['特徴量'].isin(self.selected_features)]

        # 特徴量タイプの分類
        importance_df_filtered['タイプ'] = importance_df_filtered['特徴量'].apply(
            lambda x: 'LongName TF-IDF' if x.startswith('LongName_tfidf_')
                     else 'Parent TF-IDF' if x.startswith('Parent_tfidf_')
                     else 'Project' if x.startswith('Project_')
                     else '数値'
        )

        print(f"\n=== 上位{top_n}特徴量重要度 (k-NN版、補助RandomForestから取得) ===")
        display_df = importance_df_filtered.head(top_n)[['特徴量', 'タイプ', '重要度']].copy()
        print(display_df.to_string(index=False))

        # タイプ別の統計
        print(f"\n=== 特徴量タイプ別統計 ===")
        type_stats = importance_df_filtered['タイプ'].value_counts()
        for feature_type, count in type_stats.items():
            avg_importance = importance_df_filtered[importance_df_filtered['タイプ'] == feature_type]['重要度'].mean()
            print(f"{feature_type}: {count}個 (平均重要度: {avg_importance:.4f})")

        return importance_df_filtered

    def display_knn_model_info(self):
        """k-NNモデルの詳細情報を表示"""
        if self.best_model is None:
            print("k-NNモデルが学習されていません。")
            return

        print("\n=== k-NNモデル情報 ===")
        print(f"近傍数 (k): {self.best_model.n_neighbors}")
        print(f"重み付け方法: {self.best_model.weights}")
        print(f"距離メトリック: {self.best_model.metric}")
        print(f"使用特徴量数: {len(self.selected_features) if self.selected_features else 'N/A'}")

        if self.best_params:
            print(f"\n最適化されたハイパーパラメータ:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")


# 使用例
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # k-NN版バグハンターのインスタンス作成
    bug_hunter = KNNBugHunter(
        feature_selection_threshold=0.001,
        tfidf_max_features=1000,
        java_tokenizer_min_length=3,
        include_package_tokens=False
    )

    # パイプライン実行
    results, optimal_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("k-NN版バグ予測完了!")
    print("="*60)
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")

    # k-NNモデル情報の表示
    bug_hunter.display_knn_model_info()

    # ダウンサンプリングサマリーの表示
    bug_hunter.display_downsampling_summary()

    # 特徴量重要度テーブルの表示（補助RandomForestから）
    bug_hunter.display_feature_importance_table(top_n=15)

    # トークナイザーの動作例表示
    bug_hunter.display_tokenizer_analysis(sample_size=3)

    feature_analysis = bug_hunter.get_feature_analysis()
    print(f"\n最適パラメータ: {feature_analysis['best_params']}")
    print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}")
    print(f"全特徴量数: {len(feature_analysis['all_feature_names'])}")
    print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")

    if feature_analysis['downsampling_info']:
        ds_info = feature_analysis['downsampling_info']
        print(f"データ削減率: {ds_info['reduction_rate']:.1f}%")

    print("\n" + "="*60)
