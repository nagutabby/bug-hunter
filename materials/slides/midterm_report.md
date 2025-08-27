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

鈴木研究室 笹川 尋翔

## 目次
1. 背景と関連研究
2. 目的
3. データ分析の流れ
4. 分析手法
5. 結果
6. 考察
7. 今後の見通し

# 1. 背景と関連研究
<!--
_class: lead
_paginate: false
_header: ""
-->

## 1.1 ソフトウェア開発における課題
76%の開発者がリファクタリングによるバグ混入やリグレッションの<br>リスクを感じている <sup>[1]</sup>

ソフトウェアの構造の変化に伴い、クラスや関数の理解が困難に

既存のリファクタリング手法は構造を捉えるが、変化を追跡できない

**構造の変化から、より有益な情報を得られるのではないか？**

> [1] Microsoft Research, "An Empirical Study of Refactoring Challenges and Benefits at Microsoft"

## 1.2 関連研究
- レビューコメントを収集し、保守性の低下リスクを分析<sup>[2]</sup>
- 静的解析ツールを用いてコードの改善策を提示<sup>[3]</sup>
- コードメトリクスを用いたバグ予測により、コード品質を改善<sup>[4]</sup>

> [2] X.Han et al., "Understanding Code Smell Detection via Code Review: A Study of the OpenStack Community"
> [3] S.Romano et al., "Do Static Analysis Tools Affect Software Quality when Using Test-driven Development?"'
> [4] R.Ferenc et al., "An automatically created novel bug dataset and its validation in bug prediction"

# 2. 目的
<!--
_class: lead
_paginate: false
_header: ""
-->

## 2 目的
- 時系列変化量を用いたバグ予測精度の向上
  - コードメトリクス（例: メソッドの複雑度・凝集度）の変化量を導入・分析
- バグの特徴とメトリクスの関係性の解明
  - バグの特徴を体系的に整理し、保守性とのつながりを明確化
  - バグの性質ごとにコードの改善策を示す

# 3. データ分析の流れ
<!--
_class: lead
_paginate: false
_header: ""
-->

## 3.1 ソフトウェア構造の変化の追跡
<figure style="max-width: 65vw; display: block; margin: 0 auto;">
  <img src="../images/software_structure_tracking.svg" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図1 ソフトウェア構造の変化</figcaption>
</figure>

- 比較的単純な時系列データとして、コード行数・トークン数・<br>循環的複雑度の変化量を導入

## 3.2 データセットの選定と比較データの構築
スナップショット分析のためのデータとして、BugHunter Datasetを使用

コードメトリクスの変化量をデータセットに追加

追加前と追加後で、評価指標がどの程度を改善されるかを検証

## 3.3 ランダムフォレストによるバグ予測
<figure style="max-width: 70vw; display: block; margin: 0 auto;">
  <img src="../images/random_forest_classification.svg" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図2 バグの2値分類の概要</figcaption>
</figure>

- 決定木を複数生成し、多数決で境界線を決定

# 4. 分析手法
<!--
_class: lead
_paginate: false
_header: ""
-->

## 4.1 プロジェクトの選定
<table style="font-size: 2rem; margin: 0 auto;">
  <caption>表1 選定したプロジェクト</caption>
  <thead>
    <tr>
      <th>プロジェクト</th>
      <th>役割</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>antlr4</td>
      <td>構文解析ツール</td>
    </tr>
    <tr>
      <td>BroadleafCommerce</td>
      <td>Webアプリフレームワーク</td>
    </tr>
    <tr>
      <td>ceylon-ide-eclipse</td>
      <td>IDEプラグイン</td>
    </tr>
    <tr>
      <td>elasticsearch</td>
      <td>全文検索エンジン</td>
    </tr>
    <tr>
      <td>hazelcast</td>
      <td>インメモリデータベース</td>
    </tr>
    <tr>
      <td>oryx</td>
      <td>機械学習フレームワーク</td>
    </tr>
  </tbody>
</table>

## 4.2 プロジェクトの分析
### 分析手順
1. 特徴量重要度（Feature Importance）の測定
2. ヒストグラムによる特徴量分布の確認
3. Partial Depedence Plot（PDP）による分類傾向の把握
4. 決定木の可視化による分類条件の可視化

# 5. 結果
<!--
_class: lead
_paginate: false
_header: ""
-->

## 5.1 特徴量重要度
<figure style="max-width: 65vw; display: block; margin: 0 auto;">
  <img src="../images/hazelcast/feature_importance_chart.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図3 hazelcastの特徴量重要度</figcaption>
</figure>

- トークン数・コード行数の変化量、Halsteadメトリクス、Maintainability Indexの重要度が高い

## 5.2 特徴量分布
<figure style="max-width: 75vw;　display: block; margin: 0 auto;">
  <img src="../images/ceylon-ide-eclipse/feature_histograms.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図4 ceylon-ide-eclipseの特徴量分布</figcaption>
</figure>

- 変化量やHalstead系は分散が小さく、Maintainability Indexは<br>分散が大きい

## 5.3 PDP分析
<figure style="max-width: 75vw;　display: block; margin: 0 auto;">
  <img src="../images/elasticsearch/partial_dependence_plots.png" width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図5 elasticsearchのPDP</figcaption>
</figure>

- ほとんどの特徴量において、陽性クラスの予測確率が0.5未満

## 5.4 決定木分析
<figure style="max-width: 75vw;　display: block; margin: 0 auto;">
  <img src="../images/elasticsearch/decision_tree_visualization.png"　 width="100%">
  <figcaption style="text-align: center; font-size: 2rem;">図6 elasticsearchの決定木</figcaption>
</figure>

- 陰性クラスのノードのジニ不純度が比較的低い

## 5.5 評価指標の測定
- 3つのプロジェクト: F1スコアが最大0.1向上（有意差あり）
- 2つのプロジェクト: F1スコアが向上（有意差なし）

F1スコアの伸びに伴い、適合率や再現率が全体的に向上

**モデルの予測性能が改善される傾向があることを確認**

# 6. 考察
<!--
_class: lead
_paginate: false
_header: ""
-->

## 6 考察
### 陰性クラス予測の改善
リファクタリングによるメトリクスの減少傾向を確認できた

消去法的な分類がF1スコアの改善に寄与
### さらなる精度向上に向けて
PDPや決定木を見ると、陽性クラスの予測確率が低い

バグの混入理由が多様であることが影響している可能性

# 7. 今後の見通し
<!--
_class: lead
_paginate: false
_header: ""
-->

## 7 今後の見通し
### 陽性クラスの詳細な分類による因果関係の解明
バグ混入と保守性メトリクスの関連性が不明確であるため
### バグの体系的分類によるパターンの識別
Orthogonal Defect Classification（ODC）のような分析手法を活用

バグの特徴とコードメトリクスの時系列変化の関連性を示す

## まとめ
半数のプロジェクトで有意差があり、F1スコアが最大0.1向上

コードメトリクスの変化量を組み合わせることで、より効果的な<br>バグ分類ができるようになった

今後は陽性クラスの分類精度を高め、品質改善に役立つコードの特徴を<br>捉えたい
