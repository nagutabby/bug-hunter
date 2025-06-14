from utils import BaseBugHunter, GLOBAL_SEED
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer
import pandas as pd


class SimplifiedBugHunter(BaseBugHunter):
    """
    RandomForest版バグ予測クラス（RandomUnderSampler + 相互情報量による特徴量選択使用）

    BaseBugHunterから継承し、RandomForest固有の機能を実装
    """

    def __init__(self, feature_selection_percentile: float = 30.0,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False):
        """
        コンストラクタ

        Parameters:
            feature_selection_percentile (float): 相互情報量による特徴量選択の閾値（パーセンタイル）
            tfidf_max_features (int): TF-IDFで生成する特徴量の最大数
            java_tokenizer_min_length (int): Javaトークナイザーの最小トークン長
            include_package_tokens (bool): パッケージ名のトークンを含めるかどうか
        """
        super().__init__(feature_selection_percentile, tfidf_max_features,
                        java_tokenizer_min_length, include_package_tokens)

    def evaluate_model_with_cv(self, params: dict, X_original: pd.DataFrame, y_original: pd.Series,
                              k_folds: int = 3) -> float:
        """交差検証を用いたRandomForestモデル評価（各フォールドで独立にアンダーサンプリング）"""
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

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_original, y_original)):
                # 元データから訓練・検証フォールドを作成
                X_train_fold_orig = X_original.iloc[train_idx]
                y_train_fold_orig = y_original.iloc[train_idx]
                X_val_fold = X_original.iloc[val_idx]
                y_val_fold = y_original.iloc[val_idx]

                # 訓練フォールドのみにアンダーサンプリングを適用
                X_train_fold_resampled, y_train_fold_resampled = self.apply_undersampling(
                    X_train_fold_orig, y_train_fold_orig
                )

                # モデル学習（アンダーサンプリング適用済み訓練データ）
                rf.fit(X_train_fold_resampled, y_train_fold_resampled)

                # 検証（元の検証データ）
                y_pred_proba = rf.predict_proba(X_val_fold)[:, 1]

                # Log Loss計算
                fold_loss = self.log_loss_function(y_val_fold, y_pred_proba)
                total_loss += fold_loss

                if fold_idx == 0:  # 最初のフォールドでのみログ出力
                    print(f"  フォールド例: 訓練 {len(X_train_fold_orig)}→{len(X_train_fold_resampled)}行, 検証 {len(X_val_fold)}行")

            avg_loss = total_loss / k_folds
            return avg_loss

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')

    def optimize_hyperparameters_with_log_loss(self, X_original: pd.DataFrame, y_original: pd.Series,
                                                max_iterations: int = 10) -> dict:
        """Log Loss損失関数を用いたRandomForest用ベイジアン最適化（適切な交差検証）"""
        print("\n=== RandomForest Log Lossベース ベイジアン最適化（アンダーサンプリング + 適切な交差検証）===")
        print("最適化手法: Bayesian Optimization (scikit-optimize)")
        print("探索パラメータ: n_estimators, max_depth")
        print("クラス不均衡対応: 各CVフォールドで独立にRandomUnderSampler適用")
        print("特徴量選択: 相互情報量による特徴量選択")
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

            # モデル評価（元の訓練データで交差検証、各フォールドで独立にアンダーサンプリング適用）
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

        # 参考情報として、学習済みRandomForestの特徴量重要度も取得
        if hasattr(self.best_model, 'feature_importances_'):
            rf_feature_importance = self.best_model.feature_importances_
            print("参考: 学習済みRandomForestモデルの特徴量重要度も取得しました。")

            # 上位5特徴量の表示（参考情報）
            if len(rf_feature_importance) > 0 and len(self.selected_features) == len(rf_feature_importance):
                top_indices = rf_feature_importance.argsort()[-5:][::-1]
                print("参考: RandomForest特徴量重要度 上位5:")
                for i, idx in enumerate(top_indices):
                    feature_name = self.selected_features[idx]
                    importance = rf_feature_importance[idx]
                    print(f"  {i+1}. {feature_name}: {importance:.4f}")

        return self.best_model

    def run_pipeline(self, data_path: str):
        """RandomForest用パイプライン（アンダーサンプリング + 相互情報量による特徴量選択版）"""
        print("=== RandomForest版 RandomUnderSampler + 相互情報量による特徴量選択バグ予測パイプライン ===")
        print("特徴: ")
        print("- テストデータにはアンダーサンプリングを適用せず、訓練データのみに適用")
        print("- 相互情報量による特徴量選択を使用")

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. データ分割（特徴量選択前に実施）
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_full
        )
        print(f"データ分割: 訓練データ {len(X_train)}行, テストデータ {len(X_test)}行")

        # 4. 訓練データのみにアンダーサンプリング適用
        X_train_resampled, y_train_resampled = self.apply_undersampling(X_train, y_train)
        print(f"訓練データのアンダーサンプリング: {len(X_train)}行 → {len(X_train_resampled)}行")
        print(f"テストデータ: {len(X_test)}行（アンダーサンプリング適用なし）")

        # 5. 相互情報量による特徴量選択（アンダーサンプリング適用後の訓練データで実行）
        X_train_reduced = self.select_features_by_mutual_info(X_train_resampled, y_train_resampled)

        # テストデータにも同じ特徴量選択器を適用
        X_test_reduced = self.feature_selector.transform(X_test)
        X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=self.selected_features, index=X_test.index)

        # ハイパーパラメータ最適化用に元の訓練データにも特徴量選択を適用
        X_train_reduced_orig = self.feature_selector.transform(X_train)
        X_train_reduced_orig_df = pd.DataFrame(X_train_reduced_orig, columns=self.selected_features, index=X_train.index)

        # 6. ハイパーパラメータ最適化（元の訓練データで、各CVフォールドで独立にアンダーサンプリング適用）
        optimal_params = self.optimize_hyperparameters_with_log_loss(
            X_train_reduced_orig_df, y_train, max_iterations=10
        )

        # 7. 最適化モデル学習（アンダーサンプリング適用後の訓練データで）
        optimized_model = self.train_optimized_model(
            X_train_reduced, y_train_resampled, optimal_params
        )

        # 8. 評価（元のテストデータで評価）
        results, y_pred, y_pred_proba = self.comprehensive_evaluation(
            X_test_reduced_df, y_test
        )

        return results, optimal_params


# 使用例
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # RandomForest版バグハンター（アンダーサンプリング + 相互情報量による特徴量選択使用）のインスタンス作成
    bug_hunter = SimplifiedBugHunter(
        feature_selection_percentile=30.0,
        tfidf_max_features=100,
        java_tokenizer_min_length=3,
        include_package_tokens=False
    )

    # パイプライン実行
    results, optimal_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("RandomForest版バグ予測完了!")
    print("="*60)
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"ROC-AUC: {results['ROC_AUC']:.3f}")

    # アンダーサンプリングサマリーの表示
    bug_hunter.display_sampling_summary()

    # 相互情報量による特徴量選択の結果表示
    bug_hunter.display_mutual_info_table(top_n=15)

    # 特徴量選択のサマリー表示
    bug_hunter.display_feature_selection_summary()

    # トークナイザーの動作例表示
    bug_hunter.display_tokenizer_analysis(sample_size=3)

    feature_analysis = bug_hunter.get_feature_analysis()
    print(f"\n最適パラメータ: {feature_analysis['best_params']}")
    print(f"選択された特徴量数: {len(feature_analysis['selected_features'])}")
    print(f"全特徴量数: {len(feature_analysis['all_feature_names'])}")
    print(f"特徴量選択パーセンタイル: {feature_analysis['feature_selection_percentile']}%")
    print(f"TF-IDF最大特徴量数: {feature_analysis['tfidf_max_features']}")

    if feature_analysis['sampling_info']:
        sampling_info = feature_analysis['sampling_info']
        print(f"データサイズ変化率: {sampling_info['change_rate_percent']:.1f}%")

    print("\n" + "="*60)
