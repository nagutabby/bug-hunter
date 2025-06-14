from utils import BaseBugHunter, GLOBAL_SEED
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer
import pandas as pd


class SimplifiedBugHunter(BaseBugHunter):
    """
    RandomForest版バグ予測クラス（不適切なアンダーサンプリング適用）

    BaseBugHunterから継承し、RandomForest固有の機能を実装
    """

    def __init__(self, feature_selection_threshold: float = 0.001,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False):
        """
        コンストラクタ
        """
        # k_neighborsを削除
        super().__init__(feature_selection_threshold, tfidf_max_features,
                        java_tokenizer_min_length, include_package_tokens)

    def evaluate_model_with_cv(self, params: dict, X_original: pd.DataFrame, y_original: pd.Series,
                              k_folds: int = 3) -> float:
        """
        【不適切な実装】交差検証を用いたRandomForestモデル評価
        この関数が受け取るデータは既にサンプリング済みであることを想定
        """
        try:
            # Random Forestモデルの作成
            rf = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                random_state=GLOBAL_SEED,
                n_jobs=-1
            )

            # 交差検証でLog Lossを評価
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
            total_loss = 0.0

            # ★注意：渡されるX_original, y_originalは既にサンプリング済み
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_original, y_original)):
                # 元データから訓練・検証フォールドを作成
                X_train_fold = X_original.iloc[train_idx]
                y_train_fold = y_original.iloc[train_idx]
                X_val_fold = X_original.iloc[val_idx]
                y_val_fold = y_original.iloc[val_idx]

                # ★誤り：データは事前にサンプリング済みのため、ここでは何もしない
                # モデル学習
                rf.fit(X_train_fold, y_train_fold)

                # 検証
                y_pred_proba = rf.predict_proba(X_val_fold)[:, 1]

                # Log Loss計算
                fold_loss = self.log_loss_function(y_val_fold, y_pred_proba)
                total_loss += fold_loss

            avg_loss = total_loss / k_folds
            return avg_loss

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')

    def optimize_hyperparameters_with_log_loss(self, X_original: pd.DataFrame, y_original: pd.Series,
                                                max_iterations: int = 10) -> dict:
        """Log Loss損失関数を用いたRandomForest用ベイジアン最適化"""
        print("\n=== RandomForest Log Lossベース ベイジアン最適化（事前サンプリング済みデータ使用）===")
        print("最適化手法: Bayesian Optimization (scikit-optimize)")
        print("探索パラメータ: n_estimators, max_depth")
        print("特徴量: 数値 + Java TF-IDF + One-Hot Encoding + 正規化")

        self.best_loss = float('inf')
        self.best_params = None
        self.optimization_history = []

        # 探索空間の定義
        search_space = [
            Integer(100, 1000, name='n_estimators'),
            Integer(10, 30, name='max_depth'),
        ]

        # 目的関数
        def objective(params):
            n_estimators, max_depth = params

            param_dict = {
                'n_estimators': int(n_estimators),
                'max_depth': int(max_depth),
            }

            # モデル評価（既にサンプリング済みのデータで交差検証）
            loss = self.evaluate_model_with_cv(param_dict, X_original, y_original)

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
        print("RandomForest Bayesian Optimization開始...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=max_iterations,
            random_state=GLOBAL_SEED,
            acq_func='EI',
            n_initial_points=5
        )

        print(f"\nRandomForest Bayesian Optimization完了!")
        print(f"最良損失: {self.best_loss:.4f}")
        final_best_params = self.best_params.copy() if self.best_params else {}
        print(f"最良パラメータ: {final_best_params}")
        print(f"総評価回数: {len(self.optimization_history)}")

        return final_best_params

    def train_initial_model_for_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """特徴量重要度を取得するための初期RandomForestモデルを学習"""
        print("\n=== 特徴量重要度取得のための初期RandomForestモデル学習 ===")

        initial_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        initial_rf.fit(X, y)
        self.feature_importance = initial_rf.feature_importances_
        print("初期RandomForestモデル学習と特徴量重要度の計算が完了しました。")
        return initial_rf

    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, optimal_params: dict):
        """最適化されたパラメータでRandomForestモデルを学習"""
        print("\n=== 最適化RandomForestモデル学習 ===")

        rf_params = {
            'n_estimators': int(optimal_params['n_estimators']),
            'max_depth': int(optimal_params['max_depth']),
            'random_state': GLOBAL_SEED,
            'n_jobs': -1
        }

        self.best_model = RandomForestClassifier(**rf_params)
        self.best_model.fit(X, y)

        print("最適化RandomForestモデル学習完了")
        print(f"最終パラメータ: {rf_params}")
        print(f"学習データ: {len(X)}件")

        # 特徴量重要度の取得
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
            print("学習済みRandomForestモデルから特徴量重要度を計算しました。")
        else:
            self.feature_importance = None
            print("警告: 学習済みモデルは特徴量重要度を提供しません。")

        return self.best_model

    def run_pipeline(self, data_path: str):
        """
        【不適切な実装例】RandomForest用パイプライン
        データ分割前にアンダーサンプリングを適用する誤った実装
        """
        print("="*60)
        print("### 不適切なアンダーサンプリング適用パイプライン（データリーケージあり） ###")
        print("警告: データ分割前に全体へRandomUnderSamplerを適用しています。")
        print("="*60)

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備
        X_full, y_full = self.prepare_data(data, is_training=True)
        print(f"準備完了データ: {len(X_full)}行")

        # 3. ★★★不適切な実装★★★
        # データ分割前にデータセット全体にアンダーサンプリングを適用
        print("\n★★★ 不適切な実践: データ分割前に全体へアンダーサンプリングを適用 ★★★")
        X_resampled_full, y_resampled_full = self.apply_undersampling(X_full, y_full)
        print(f"データセット全体のアンダーサンプリング: {len(X_full)}行 → {len(X_resampled_full)}行")
        print("これによりテストデータが訓練データから削除されるべき多数派クラスの選別に影響を与え、一種のリーケージが発生します。")

        # 4. データ分割（アンダーサンプリング済みのデータに対して実施）
        # これにより、テストセットの評価が本来よりも不当に変動する可能性がある
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled_full, y_resampled_full, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_resampled_full
        )
        print(f"\nアンダーサンプリング済みデータからの分割: 訓練 {len(X_train)}行, テスト {len(X_test)}行")

        # 5. 特徴量重要度取得のための初期モデル学習（既にサンプリング済みの訓練データを使用）
        self.train_initial_model_for_feature_importance(X_train, y_train)

        # 6. 特徴量削減
        X_train_reduced = self.select_features_by_importance(X_train)
        X_test_reduced = self.select_features_by_importance(X_test)

        # 7. ハイパーパラメータ最適化（既にサンプリング済みの訓練データを使用）
        # ★注意：evaluate_model_with_cvも内部でサンプリングをしないように修正済み
        optimal_params = self.optimize_hyperparameters_with_log_loss(
            X_train_reduced, y_train, max_iterations=10
        )

        # 8. 最適化モデル学習（既にサンプリング済みの訓練データで）
        optimized_model = self.train_optimized_model(
            X_train_reduced, y_train, optimal_params
        )

        # 9. 評価
        # これにより評価指標が人為的に高く、または低く算出される可能性がある
        results, y_pred, y_pred_proba = self.comprehensive_evaluation(
            X_test_reduced, y_test
        )

        return results, optimal_params


# 使用例
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # RandomForest版バグハンターのインスタンス作成
    bug_hunter = SimplifiedBugHunter(
        feature_selection_threshold=0.001,
        tfidf_max_features=100,
        java_tokenizer_min_length=3,
        include_package_tokens=False
        # k_neighborsを削除
    )

    # パイプライン実行
    results, optimal_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("【不適切な】RandomForest版バグ予測完了!")
    print("="*60)
    print("警告: 以下の評価指標はデータリーケージにより信頼できない可能性があります。")
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"ROC-AUC: {results['ROC_AUC']:.3f}")

    # アンダーサンプリングサマリーの表示
    bug_hunter.display_sampling_summary()

    # 特徴量重要度テーブルの表示
    bug_hunter.display_feature_importance_table(top_n=15)

    # トークナイザーの動作例表示
    bug_hunter.display_tokenizer_analysis(sample_size=3)

    feature_analysis = bug_hunter.get_feature_analysis()
    print(f"\n最適パラメータ: {feature_analysis['best_params']}")
    print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}")
    print(f"全特徴量数: {len(feature_analysis['all_feature_names'])}")
    print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")

    if feature_analysis['sampling_info']:
        sampling_info = feature_analysis['sampling_info']
        print(f"データサイズ変化率: {sampling_info['change_rate_percent']:.1f}%")

    print("\n" + "="*60)
