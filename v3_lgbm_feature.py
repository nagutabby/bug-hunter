import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from v2_lgbm import LightGBMBugHunter

import warnings
warnings.filterwarnings('ignore')

class SoftwareMetricsFeatureEngineer:
    """
    ソフトウェアメトリクスに特化した特徴量エンジニアリングクラス
    相互情報量の結果に基づいて効果的な特徴量を生成
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    def create_halstead_ratios(self, df):
        """
        Halsteadメトリクス系の比率・組み合わせ特徴量を作成
        上位のHNDB, HEFF, HCPLなどを活用
        """
        new_features = df.copy()

        # 基本的なHalstead比率
        with np.errstate(divide='ignore', invalid='ignore'):
            # 効率性比率（HEFF関連）
            new_features['HEFF_HNDB_ratio'] = np.where(df['HNDB'] != 0, df['HEFF'] / df['HNDB'], 0)
            new_features['HEFF_HTRP_ratio'] = np.where(df['HTRP'] != 0, df['HEFF'] / df['HTRP'], 0)

            # 複雑性比率（HCPL関連）
            new_features['HCPL_HVOL_ratio'] = np.where(df['HVOL'] != 0, df['HCPL'] / df['HVOL'], 0)
            new_features['HCPL_HDIF_ratio'] = np.where(df['HDIF'] != 0, df['HCPL'] / df['HDIF'], 0)

            # プログラム長関連
            new_features['HPL_HPV_ratio'] = np.where(df['HPV'] != 0, df['HPL'] / df['HPV'], 0)
            new_features['HPL_HVOL_ratio'] = np.where(df['HVOL'] != 0, df['HPL'] / df['HVOL'], 0)

            # 理論的長さ vs 実際の長さ
            new_features['HTRP_HPL_ratio'] = np.where(df['HPL'] != 0, df['HTRP'] / df['HPL'], 0)

        return new_features

    def create_maintainability_features(self, df):
        """
        Maintainability Index系の特徴量を作成
        MI, MIMS, MISEI, MISMが上位にランクイン
        """
        new_features = df.copy()

        with np.errstate(divide='ignore', invalid='ignore'):
            # MI系の比率
            new_features['MI_MIMS_ratio'] = np.where(df['MIMS'] != 0, df['MI'] / df['MIMS'], 0)
            new_features['MISEI_MISM_ratio'] = np.where(df['MISM'] != 0, df['MISEI'] / df['MISM'], 0)

            # 複合メンテナンス指標
            new_features['MI_complexity_score'] = df['MI'] * df['MIMS'] / (df['MISM'] + 1)
            new_features['MISEI_efficiency'] = df['MISEI'] / (df['MISM'] + df['MIMS'] + 1)

            # MI系の統合スコア
            new_features['MI_total_score'] = (df['MI'] + df['MIMS'] + df['MISEI']) / 3
            new_features['MI_weighted_score'] = (df['MI'] * 0.4 + df['MIMS'] * 0.3 + df['MISEI'] * 0.3)

        return new_features

    def create_complexity_features(self, df):
        """
        複雑性関連の特徴量を作成
        LOC, LLOC, CCなどを活用
        """
        new_features = df.copy()

        with np.errstate(divide='ignore', invalid='ignore'):
            # コード密度
            new_features['code_density'] = np.where(df['LOC'] != 0, df['LLOC'] / df['LOC'], 0)
            new_features['comment_density'] = np.where(df['LOC'] != 0, df['CLOC'] / df['LOC'], 0)

            # 複雑性とサイズの関係
            new_features['CC_per_LOC'] = np.where(df['LOC'] != 0, df['CC'] / df['LOC'], 0)
            new_features['McCC_per_LLOC'] = np.where(df['LLOC'] != 0, df['McCC'] / df['LLOC'], 0)

            # ネストの複雑性
            new_features['nested_complexity'] = df['CC'] * df['McCC'] / (df['LOC'] + 1)

            # パラメータ複雑性
            new_features['param_complexity'] = df['NUMPAR'] * df['CC'] / (df['LOC'] + 1)

        return new_features

    def create_warning_features(self, df):
        """
        警告系の特徴量を作成
        警告レベルの重み付け合計など
        """
        new_features = df.copy()

        # 警告の重み付け合計（重要度に応じて重み付け）
        warning_weights = {
            'WarningBlocker': 5,
            'WarningCritical': 4,
            'WarningMajor': 3,
            'WarningMinor': 2,
            'WarningInfo': 1
        }

        new_features['weighted_warnings'] = 0
        for warning_type, weight in warning_weights.items():
            if warning_type in df.columns:
                new_features['weighted_warnings'] += df[warning_type] * weight

        # 警告密度
        with np.errstate(divide='ignore', invalid='ignore'):
            total_warnings = sum(df[col] for col in warning_weights.keys() if col in df.columns)
            new_features['warning_density'] = np.where(df['LOC'] != 0, total_warnings / df['LOC'], 0)

        # 高レベル警告の比率
        high_level_warnings = df.get('WarningBlocker', 0) + df.get('WarningCritical', 0)
        new_features['high_level_warning_ratio'] = np.where(
            total_warnings != 0, high_level_warnings / total_warnings, 0
        )

        return new_features

    def create_rule_features(self, df):
        """
        PMDルール系の特徴量を作成
        ルールカテゴリの組み合わせ
        """
        new_features = df.copy()

        # ルール系の列を特定
        rule_columns = [col for col in df.columns if 'Rules' in col]

        if rule_columns:
            # 総ルール違反数
            new_features['total_rule_violations'] = df[rule_columns].sum(axis=1)

            # アクティブルール数（違反があるルールの数）
            new_features['active_rule_count'] = (df[rule_columns] > 0).sum(axis=1)

            # ルール違反密度
            with np.errstate(divide='ignore', invalid='ignore'):
                new_features['rule_violation_density'] = np.where(
                    df['LOC'] != 0, new_features['total_rule_violations'] / df['LOC'], 0
                )

            # 重要ルールの特徴量（経験的に重要とされるルール）
            important_rules = ['Design Rules', 'Code Size Rules', 'Coupling Rules',
                             'Unnecessary and Unused Code Rules', 'Security Code Guideline Rules']

            important_rule_cols = [col for col in important_rules if col in df.columns]
            if important_rule_cols:
                new_features['important_rule_violations'] = df[important_rule_cols].sum(axis=1)

        return new_features

    def create_polynomial_features(self, df, target_features):
        """
        重要な特徴量の2次多項式特徴量を作成
        """
        # 上位の特徴量のみを対象
        target_df = df[target_features].copy()

        # 2次の交互作用項のみを作成
        poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_array = poly_features.fit_transform(target_df)

        # 新しい特徴量の名前を作成
        feature_names = poly_features.get_feature_names_out(target_features)

        # 元の特徴量を除外（交互作用項のみを残す）
        interaction_indices = [i for i, name in enumerate(feature_names)
                             if ' ' in name]  # 交互作用項には空白が含まれる

        if interaction_indices:
            interaction_features = poly_array[:, interaction_indices]
            interaction_names = [feature_names[i] for i in interaction_indices]

            # DataFrameに変換
            poly_df = pd.DataFrame(interaction_features,
                                 columns=interaction_names,
                                 index=df.index)

            return pd.concat([df, poly_df], axis=1)

        return df

    def create_statistical_features(self, df):
        """
        統計的特徴量を作成
        """
        new_features = df.copy()

        # 主要メトリクスの統計量
        main_metrics = ['HNDB', 'HEFF', 'HTRP', 'MISM', 'MISEI', 'MIMS', 'MI']

        if all(col in df.columns for col in main_metrics):
            metrics_df = df[main_metrics]

            # 行ごとの統計量
            new_features['metrics_mean'] = metrics_df.mean(axis=1)
            new_features['metrics_std'] = metrics_df.std(axis=1)
            new_features['metrics_max'] = metrics_df.max(axis=1)
            new_features['metrics_min'] = metrics_df.min(axis=1)
            new_features['metrics_range'] = new_features['metrics_max'] - new_features['metrics_min']

            # 分散・歪度・尖度
            new_features['metrics_var'] = metrics_df.var(axis=1)
            new_features['metrics_skew'] = metrics_df.skew(axis=1)
            new_features['metrics_kurtosis'] = metrics_df.kurtosis(axis=1)

        return new_features

    def create_log_features(self, df):
        """
        対数変換特徴量を作成（正の値のみ）
        """
        new_features = df.copy()

        # 対数変換に適した特徴量
        log_candidates = ['HVOL', 'LOC', 'LLOC', 'TNOS', 'HPL', 'HPV']

        for col in log_candidates:
            if col in df.columns:
                # 正の値のみ対数変換
                positive_mask = df[col] > 0
                new_features[f'{col}_log'] = 0.0
                new_features.loc[positive_mask, f'{col}_log'] = np.log1p(df.loc[positive_mask, col])

        return new_features

    def apply_feature_engineering(self, df):
        """
        すべての特徴量エンジニアリングを適用
        """
        print("特徴量エンジニアリング実行中...")

        # 元の特徴量数
        original_features = df.shape[1]

        # 1. Halstead関連特徴量
        df = self.create_halstead_ratios(df)
        print(f"Halstead特徴量追加: {df.shape[1] - original_features}個")

        # 2. Maintainability関連特徴量
        df = self.create_maintainability_features(df)
        print(f"MI特徴量追加: {df.shape[1] - original_features}個")

        # 3. 複雑性関連特徴量
        df = self.create_complexity_features(df)
        print(f"複雑性特徴量追加: {df.shape[1] - original_features}個")

        # 4. 警告関連特徴量
        df = self.create_warning_features(df)
        print(f"警告特徴量追加: {df.shape[1] - original_features}個")

        # 5. ルール関連特徴量
        df = self.create_rule_features(df)
        print(f"ルール特徴量追加: {df.shape[1] - original_features}個")

        # 6. 統計的特徴量
        df = self.create_statistical_features(df)
        print(f"統計特徴量追加: {df.shape[1] - original_features}個")

        # 7. 対数変換特徴量
        df = self.create_log_features(df)
        print(f"対数特徴量追加: {df.shape[1] - original_features}個")

        # 8. 重要特徴量の多項式特徴量（交互作用項のみ）
        top_features = ['HNDB', 'HEFF', 'HTRP', 'MISM', 'MISEI', 'MIMS', 'MI', 'HVOL']
        available_top_features = [f for f in top_features if f in df.columns]

        if len(available_top_features) >= 2:
            df = self.create_polynomial_features(df, available_top_features)
            print(f"多項式特徴量追加: {df.shape[1] - original_features}個")

        # 無限値・NaN値の処理
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        print(f"特徴量エンジニアリング完了: {original_features} → {df.shape[1]}個")

        return df

    def get_feature_importance_analysis(self, df, target_col='Number of Bugs'):
            """
            新しく作成した特徴量の重要度分析
            """
            if target_col not in df.columns:
                print(f"警告: ターゲット列 '{target_col}' が見つかりません")
                return None

            from sklearn.feature_selection import mutual_info_classif

            # ターゲットをバイナリに変換
            y = (df[target_col] > 0).astype(int)
            X = df.drop(columns=[target_col])

            # 数値列のみを選択
            X_numeric = X.select_dtypes(include=np.number)


            # 相互情報量を計算
            mi_scores = mutual_info_classif(X_numeric, y, random_state=42)

            # 結果をDataFrameに整理
            feature_importance = pd.DataFrame({
                'feature': X_numeric.columns,
                'mutual_info_score': mi_scores
            }).sort_values('mutual_info_score', ascending=False)

            print("\n=== 新しい特徴量の相互情報量スコア（上位20） ===")
            print(feature_importance.head(20).to_string(index=False))

            return feature_importance


# 使用例とLightGBMBugHunterクラスの拡張
class EnhancedLightGBMBugHunter(LightGBMBugHunter):
    """
    特徴量エンジニアリングを組み込んだLightGBMBugHunter
    """

    def __init__(self, feature_selection_percentile: float = 30.0,
                 tfidf_max_features: int = 100,
                 java_tokenizer_min_length: int = 2,
                 include_package_tokens: bool = False,
                 apply_feature_engineering: bool = True):
        super().__init__(feature_selection_percentile, tfidf_max_features,
                        java_tokenizer_min_length, include_package_tokens)
        self.apply_feature_engineering = apply_feature_engineering
        self.feature_engineer = SoftwareMetricsFeatureEngineer()

    def prepare_data(self, data, is_training=True):
        """
        データ準備処理に特徴量エンジニアリングを組み込み
        """
        # 特徴量エンジニアリングを適用
        if self.apply_feature_engineering:
            print("\n=== 特徴量エンジニアリング適用 ===")
            data = self.feature_engineer.apply_feature_engineering(data)

            # 新しい特徴量の分析
            if is_training and 'Number of Bugs' in data.columns:
                self.feature_engineer.get_feature_importance_analysis(data)

        # 親クラスのprepare_data を呼び出し
        return super().prepare_data(data, is_training)


# 使用例
if __name__ == "__main__":
    # 拡張版の使用例
    print("=== 特徴量エンジニアリング拡張版LightGBMBugHunter ===")

    # CSVファイルパスを指定
    data_path = "method-p.csv"

    # 拡張版バグハンターのインスタンス作成
    enhanced_bug_hunter = EnhancedLightGBMBugHunter(
        feature_selection_percentile=30.0,
        tfidf_max_features=100,
        java_tokenizer_min_length=3,
        include_package_tokens=False,
        apply_feature_engineering=True
    )

    # パイプライン実行
    results, optimal_params = enhanced_bug_hunter.run_pipeline(data_path)

    print("\n" + "="*60)
    print("拡張版LightGBMバグ予測完了!")
    print("="*60)
    print(f"F1スコア: {results['F1']:.3f}")
    print(f"Precision: {results['Precision']:.3f}")
    print(f"Recall: {results['Recall']:.3f}")
    print(f"Accuracy: {results['Accuracy']:.3f}")
    print(f"ROC-AUC: {results['ROC_AUC']:.3f}")

    # 追加の分析結果表示
    enhanced_bug_hunter.display_class_weight_summary()
    enhanced_bug_hunter.display_mutual_info_table(top_n=20)
    enhanced_bug_hunter.display_feature_selection_summary()
