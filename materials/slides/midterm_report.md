---
marp: true
size: 16:9
paginate: true
theme: academic
header: 中間発表
headingDivider: 2
---


# コードの時系列変化を考慮した<br>保守性低下の要因分析と改善
<!--
_class: lead
_paginate: false
_header: ""
-->

鈴木研究室 2410064
笹川 尋翔

## 目次
1. 背景
2. 目的
3. 関連研究
4. 分析手順
5. 結果
6. 考察
7. 課題と展望

# 1. 背景
<!--
_class: lead
_paginate: false
-->

## 1. 背景
<figure style="max-width: 50vw; display: block; margin: 0 auto;">
  <img src="../images/module_hierarchy.svg" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図1 ソフトウェア構造の複雑化</figcaption>
</figure>

76%の開発者がリファクタリングによるバグ混入のリスクを認識 <sup>[1]</sup>

ソフトウェア構造の変化に伴い、品質保証とバグの特定が困難に

> [1] Microsoft Research, "An Empirical Study of Refactoring Challenges and Benefits at Microsoft",　2014

# 2. 目的
<!--
_class: lead
_paginate: false
-->

## 2. 目的
- 保守性指標の時系列変化量を品質改善に役立てる
  - これまで考慮されていなかった保守性の変化に着目
  - 例: メソッドの複雑度の変化、テストやドキュメント数の変化
- 開発プロセスとコードメトリクスの関係性を解明
  - コード自体の変化と開発者の行動を対応付け、具体的な改善策を提示

# 3. 関連研究
<!--
_class: lead
_paginate: false
-->

## 3. 関連研究
- レビューコメントを収集し、保守性の低下リスクを分析<sup>[2]</sup>
- 静的解析ツールを用いてコードの改善策を提示<sup>[3]</sup>
- コードメトリクスを用いたバグ予測により、コード品質を改善<sup>[4]</sup>

> [2] X.Han et al., "Understanding Code Smell Detection via Code Review: A Study of the OpenStack Community", 2021
> [3] S.Romano et al., "Do Static Analysis Tools Affect Software Quality when Using Test-driven Development?", 2022
> [4] R.Ferenc et al., "An automatically created novel bug dataset and its validation in bug prediction", 2020

# 4. 分析手順
<!--
_class: lead
_paginate: false
-->

## 4. 分析手順
<figure style="max-width: 55vw; display: block; margin: 0 auto;">
  <img src="../images/data_analysis_flow.svg" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図2 データ分析の流れ</figcaption>
</figure>

1. 時系列変化量を追加
2. テキストデータを数値に変換
3. 特徴量の追加前・追加後のデータを用いて機械学習モデルを訓練
4. 各プロジェクトでモデルの性能を評価


## 4.1 時系列変化量の追加
<figure style="max-width: 35vw; display: block; margin: 0 auto;">
  <img src="../images/software_structure_tracking.svg" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図3 ソフトウェア構造の変化</figcaption>
</figure>

- 単純な時系列データとして、コード行数・トークン数・循環的複雑度の変化量をデータセットに追加

## 4.2　テキストデータの数値変換
- クラスやメソッドの識別子を単語ごとに分解
  - 単語の出現率などの類似性に基づいて学習できるようにする
- メソッドの変更履歴（例: 新規追加、変更、削除）を表す変数を導入
  - 変化量が欠損した理由を説明できるようにする

## 4.3 機械学習モデルの訓練
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; align-items: end; width: 110vw; margin: 0 auto;">
  <figure style="margin: 0;">
    <img src="../images/decision_tree.svg" width="100%">
    <figcaption style="text-align: center; font-size: 2rem;">図4 複数の決定木を用いた投票</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="../images/random_forest_classification.svg" width="100%">
    <figcaption style="text-align: center; font-size: 2rem;">図5 決定境界による2値分類</figcaption>
  </figure>
</div>

- 決定木を複数生成し、多数決で境界線を決定

## 4.4 プロジェクトごとの性能評価
<table style="font-size: 2rem; margin: 0 auto;">
  <caption>表1 選定したプロジェクト</caption>
  <thead>
    <tr>
      <th>プロジェクト</th>
      <th>コード行数<br>（概算）</th>
      <th>コミット数</th>
      <th>バグレポートの数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ANTLR v4</td>
      <td>68,000</td>
      <td>6,526</td>
      <td>179</td>
    </tr>
    <tr>
      <td>Broadleaf Commerce</td>
      <td>322,000</td>
      <td>14,920</td>
      <td>703</td>
    </tr>
    <tr>
      <td>Eclipse plugin for Ceylon</td>
      <td>181,000</td>
      <td>7,984</td>
      <td>923</td>
    </tr>
    <tr>
      <td>Elasticsearch</td>
      <td>995,000</td>
      <td>28,815</td>
      <td>4,494</td>
    </tr>
    <tr>
      <td>Hazelcast</td>
      <td>949,000</td>
      <td>24,380</td>
      <td>3,882</td>
    </tr>
    <tr>
      <td>Oryx</td>
      <td>34,000</td>
      <td>1,054</td>
      <td>67</td>
    </tr>
  </tbody>
</table>

## 4.4 プロジェクトごとの性能評価
1. 特徴量重要度（Feature Importance）を算出
2. ヒストグラムで特徴量分布を確認
3. Partial Depedence Plot（PDP）を用いて分類傾向を把握
4. 決定木で分類の流れを可視化し、判断の基準と信頼性を確認

# 5. 結果
<!--
_class: lead
_paginate: false
-->

## 5.1 特徴量重要度
<figure style="max-width: 35vw; display: block; margin: 0 auto;">
  <img src="../images/hazelcast/feature_importance_chart.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図6 Hazelcastの特徴量重要度</figcaption>
</figure>

- トークン数・コード行数の変化量、Halsteadメトリクス、Maintainability Indexの重要度が高い

## 5.2 特徴量分布
<figure style="max-width: 45vw;　display: block; margin: 0 auto;">
  <img src="../images/ceylon-ide-eclipse/feature_histograms.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図7 Eclipse plugin for Ceylonの特徴量分布</figcaption>
</figure>

- 変化量やHalstead系は分散が小さく、Maintainability Indexは<br>分散が大きい

## 5.3 PDP分析
<figure style="max-width: 45vw;　display: block; margin: 0 auto;">
  <img src="../images/elasticsearch/partial_dependence_plots.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図8 ElasticsearchのPDP</figcaption>
</figure>

- ほとんどの特徴量において、陽性クラスの予測確率が0.5未満

## 5.4 決定木分析
<figure style="max-width: 45vw;　display: block; margin: 0 auto;">
  <img src="../images/elasticsearch/decision_tree_visualization.png"　 width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図9 Elasticsearchの決定木</figcaption>
</figure>

- 陰性クラスのノードのジニ不純度が比較的低い

## 5.5 評価指標の測定
<table style="font-size: 2rem; margin: 0 auto;">
  <caption>表2 評価指標の値の変化</caption>
  <thead>
    <tr>
      <th>プロジェクト</td>
      <th>F1スコア (変更前)</td>
      <th>F1スコア (変更後)</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Eclipse plugin for Ceylon</td>
      <td>0.39</td>
      <td>0.49</td>
    </tr>
    <tr>
      <td>Elasticsearch</td>
      <td>0.62</td>
      <td>0.72</td>
    </tr>
    <tr>
      <td>Hazelcast</td>
      <td>0.67</td>
      <td>0.71</td>
    </tr>
    <tr>
      <td>ANTLR v4</td>
      <td>0.45</td>
      <td>0.48</td>
    </tr>
    <tr>
      <td>Oryx</td>
      <td>0.33</td>
      <td>0.39</td>
    </tr>
  </tbody>
</table>

モデルの予測性能が改善される傾向があることを確認

# 6. 考察
<!--
_class: lead
_paginate: false
-->

## 6 考察
### 陰性クラス予測の改善
リファクタリングによるメトリクスの減少傾向を確認できた

消去法的な分類がF1スコアの改善に寄与
### さらなる精度向上に向けて
PDPや決定木を見ると、陽性クラスの予測確率が低い

バグの混入理由が多様であることが影響している可能性

# 7. 課題と展望
<!--
_class: lead
_paginate: false
-->

## 7 課題と展望
### 開発プロセスの理解
レビュー記録や自動テストの内容からバグが生じる状況を説明
### 陽性クラスの詳細な分類による因果関係の解明
開発プロセスやコードメトリクスの変化がバグにどのように影響するかを<br>明らかにする

## まとめ
半数のプロジェクトで有意差があり、F1スコアが最大0.1向上

コードメトリクスの変化量を組み合わせることで、より効果的な<br>バグ分類ができるようになった

今後は陽性クラスの分類精度を高め、品質改善に役立つ特徴を探す
