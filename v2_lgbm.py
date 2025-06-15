from utils import BaseBugHunter, GLOBAL_SEED
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real
import pandas as pd
import numpy as np


class LightGBMBugHunter(BaseBugHunter):
    """
    LightGBM版バグ予測クラス（scale_pos_weight + 相互情報量による特徴量選択使用）

    BaseBugHunterから継承し、LightGBM固有の機能を実装
    アンダーサンプリングの代わりにscale_pos_weightでクラス不均衡に対応
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
        self.scale_pos_weight = None

    def calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """クラス不均衡に対応するscale_pos_weightを計算"""
        # 負例の数 / 正例の数
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()

        if pos_count == 0:
            return 1.0

        scale_pos_weight = neg_count / pos_count
        print(f"クラス比率 - 負例: {neg_count}, 正例: {pos_count}")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")

        return scale_pos_weight

    def evaluate_model_with_cv(self, params: dict, X_original: pd.DataFrame, y_original: pd.Series,
                              k_folds: int = 3) -> float:
        """交差検証を用いたLightGBMモデル評価（scale_pos_weight使用）"""
        try:
            # 交差検証でLog Lossを評価
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
            total_loss = 0.0

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_original, y_original)):
                # 元データから訓練・検証フォールドを作成
                X_train_fold = X_original.iloc[train_idx]
                y_train_fold = y_original.iloc[train_idx]
                X_val_fold = X_original.iloc[val_idx]
                y_val_fold = y_original.iloc[val_idx]

                # 訓練フォールドのscale_pos_weightを計算
                fold_scale_pos_weight = self.calculate_scale_pos_weight(y_train_fold)

                # LightGBMモデルの作成
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=int(params['n_estimators']),
                    max_depth=int(params['max_depth']),
                    learning_rate=params['learning_rate'],
                    num_leaves=int(params['num_leaves']),
                    scale_pos_weight=fold_scale_pos_weight,
                    random_state=GLOBAL_SEED,
                    objective='binary',
                    metric='binary_logloss',
                    verbosity=-1,
                    n_jobs=-1
                )

                # モデル学習
                lgb_model.fit(X_train_fold, y_train_fold)

                # 検証
                y_pred_proba = lgb_model.predict_proba(X_val_fold)[:, 1]

                # Log Loss計算
                fold_loss = self.log_loss_function(y_val_fold, y_pred_proba)
                total_loss += fold_loss

                if fold_idx == 0:  # 最初のフォールドでのみログ出力
                    print(f"  フォールド例: 訓練 {len(X_train_fold)}行, 検証 {len(X_val_fold)}行")
                    print(f"  scale_pos_weight: {fold_scale_pos_weight:.2f}")

            avg_loss = total_loss / k_folds
            return avg_loss

        except Exception as e:
            print(f"評価エラー: {e}")
            return float('inf')

    def optimize_hyperparameters_with_log_loss(self, X_original: pd.DataFrame, y_original: pd.Series,
                                                max_iterations: int = 15) -> dict:
        """Log Loss損失関数を用いたLightGBM用ベイジアン最適化（scale_pos_weight使用）"""
        print("\n=== LightGBM Log Lossベース ベイジアン最適化（scale_pos_weight + 相互情報量による特徴量選択）===")
        print("最適化手法: Bayesian Optimization (scikit-optimize)")
        print("探索パラメータ: n_estimators, max_depth, learning_rate, num_leaves")
        print("クラス不均衡対応: scale_pos_weight（アンダーサンプリングなし）")
        print("特徴量選択: 相互情報量による特徴量選択")
        print("特徴量: 数値 + Java TF-IDF + One-Hot Encoding + 正規化")

        self.best_loss = float('inf')
        self.best_params = None
        self.optimization_history = []

        # 探索空間の定義
        search_space = [
            Integer(100, 1000, name='n_estimators'),
            Integer(5, 30, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Integer(10, 300, name='num_leaves'),
        ]

        # 目的関数
        def objective(params):
            n_estimators, max_depth, learning_rate, num_leaves = params

            param_dict = {
                'n_estimators': int(n_estimators),
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'num_leaves': int(num_leaves),
            }

            # モデル評価（交差検証）
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
        print("LightGBM Bayesian Optimization開始...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=max_iterations,
            random_state=GLOBAL_SEED,
            acq_func='EI',
            n_initial_points=7
        )

        print(f"\nLightGBM Bayesian Optimization完了!")
        print(f"最良損失: {self.best_loss:.4f}")
        final_best_params = self.best_params.copy() if self.best_params else {}
        print(f"最良パラメータ: {final_best_params}")
        print(f"総評価回数: {len(self.optimization_history)}")

        return final_best_params

    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, optimal_params: dict):
        """最適化されたパラメータでLightGBMモデルを学習"""
        print("\n=== 最適化LightGBMモデル学習 ===")

        # 全訓練データのscale_pos_weightを計算
        self.scale_pos_weight = self.calculate_scale_pos_weight(y)

        lgb_params = {
            'n_estimators': int(optimal_params['n_estimators']),
            'max_depth': int(optimal_params['max_depth']),
            'learning_rate': optimal_params['learning_rate'],
            'num_leaves': int(optimal_params['num_leaves']),
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': GLOBAL_SEED,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_jobs': -1
        }

        self.best_model = lgb.LGBMClassifier(**lgb_params)
        self.best_model.fit(X, y)

        print("最適化LightGBMモデル学習完了")
        print(f"最終パラメータ: {lgb_params}")
        print(f"学習データ: {len(X)}件")

        # 参考情報として、学習済みLightGBMの特徴量重要度も取得
        if hasattr(self.best_model, 'feature_importances_'):
            lgb_feature_importance = self.best_model.feature_importances_
            print("参考: 学習済みLightGBMモデルの特徴量重要度も取得しました。")

            # 上位5特徴量の表示（参考情報）
            if len(lgb_feature_importance) > 0 and len(self.selected_features) == len(lgb_feature_importance):
                top_indices = lgb_feature_importance.argsort()[-5:][::-1]
                print("参考: LightGBM特徴量重要度 上位5:")
                for i, idx in enumerate(top_indices):
                    feature_name = self.selected_features[idx]
                    importance = lgb_feature_importance[idx]
                    print(f"  {i+1}. {feature_name}: {importance:.4f}")

        return self.best_model

    def run_pipeline(self, data_path: str):
        """LightGBM用パイプライン（scale_pos_weight + 相互情報量による特徴量選択版）"""
        print("=== LightGBM版 scale_pos_weight + 相互情報量による特徴量選択バグ予測パイプライン ===")
        print("特徴: ")
        print("- アンダーサンプリングの代わりにscale_pos_weightでクラス不均衡に対応")
        print("- 相互情報量による特徴量選択を使用")
        print("- 全データを使用してモデル学習")

        # 1. データ読み込み
        data = self.read_data(data_path)

        # 2. データ準備
        X_full, y_full = self.prepare_data(data, is_training=True)

        # 3. データ分割（特徴量選択前に実施）
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_full
        )
        print(f"データ分割: 訓練データ {len(X_train)}行, テストデータ {len(X_test)}行")

        # クラス分布を確認
        print(f"訓練データのクラス分布: 負例 {(y_train == 0).sum()}, 正例 {(y_train == 1).sum()}")
        print(f"テストデータのクラス分布: 負例 {(y_test == 0).sum()}, 正例 {(y_test == 1).sum()}")

        # 4. 相互情報量による特徴量選択（元の訓練データで実行）
        X_train_reduced = self.select_features_by_mutual_info(X_train, y_train)

        # テストデータにも同じ特徴量選択器を適用
        X_test_reduced = self.feature_selector.transform(X_test)
        X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=self.selected_features, index=X_test.index)

        # 5. ハイパーパラメータ最適化（元の訓練データで交差検証）
        optimal_params = self.optimize_hyperparameters_with_log_loss(
            X_train_reduced, y_train, max_iterations=15
        )

        # 6. 最適化モデル学習（全訓練データで）
        optimized_model = self.train_optimized_model(
            X_train_reduced, y_train, optimal_params
        )

        # 7. 評価（テストデータで評価）
        results, y_pred, y_pred_proba = self.comprehensive_evaluation(
            X_test_reduced_df, y_test
        )

        return results, optimal_params

    def display_class_weight_summary(self):
        """クラス重み付けのサマリー表示"""
        if self.scale_pos_weight is not None:
            print("\n=== クラス重み付けサマリー ===")
            print(f"scale_pos_weight: {self.scale_pos_weight:.2f}")
            print("（正例に対する重み付け：負例数/正例数）")
            print("アンダーサンプリングの代わりにscale_pos_weightでクラス不均衡に対応")


# 使用例
if __name__ == "__main__":
    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # LightGBM版バグハンター（scale_pos_weight + 相互情報量による特徴量選択使用）のインスタンス作成
    bug_hunter = LightGBMBugHunter(
        feature_selection_percentile=30.0,
        tfidf_max_features=100,
        java_tokenizer_min_length=3,
        include_package_tokens=False
    )

    # パイプライン実行
    results, optimal_params = bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("LightGBM版バグ予測完了!")
    print("="*60)
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"ROC-AUC: {results['ROC_AUC']:.3f}")

    # クラス重み付けサマリーの表示
    bug_hunter.display_class_weight_summary()

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

    print("\n" + "="*60)
