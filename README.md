# Quarter-resolution Discrete Flow Maps segmentation

Cityscapes（19 semantic + void の20 state）とADE20K（ignore 0を含む151
state）向けの独立したDFM実装です。生成状態は入力画像と同じ解像度ではなく、
`model.state_downsample_factor`（標準4）から動的に決まる `ceil(H/4) ×
ceil(W/4)` です。通常の学習・評価入力はsize divisorで4の倍数なので、これは
厳密に `H/4 × W/4` になります。

Stage 1、Stage 2、joint training、PSD/CSD/ECLD/ESD、single GPU、単一ノード
DDP、bf16 AMP、JVP、gradient accumulation/clipping、checkpoint resumeを維持
しています。元の参照実装 `/home/igarashi_25/playground_2/CSDFM/DFM` は変更しません。

## 1/4 state設計

生成に関係する処理はすべてstate空間で行います。

| tensor / 処理 | ADE20K 512×512 | 可変入力 512×1024 | 補間 |
|---|---:|---:|---|
| image | 512×512 | 512×1024 | datasetの既存仕様 |
| source `mu/logvar/x0` | 128×128 | 128×256 | continuous: bilinear |
| `target_state/one_hot_state` | 128×128 | 128×256 | GT: nearest |
| `xs/xu/xt/x1` | 128×128 | 128×256 | state空間 |
| RRDB image feature | 128×128 | 128×256 | encoder内stride-2×2 |
| endpoint logits | 128×128 | 128×256 | state空間 |
| PSD/CSD/ECLD/ESD | 128×128 | 128×256 | resizeしない |
| primary/source supervision | 512×512 | 512×1024 | logits/muをbilinear拡大 |
| final evaluation | 512×512 | 512×1024 | continuousを拡大後argmax |

Cityscapesの代表例は、256×512入力に対してstateが64×128、512×1024入力に
対して128×256です。旧実装ではsource、image feature、DFM state、endpointが
すべて入力と同じ解像度でした。新実装ではloss/evaluationだけをfull resolutionへ
戻します。

### Source Generator

SegFormerの1/4 featureを基準にし、1/8、1/16、1/32 featureをbilinear
resizeしてconcatします。軽量UNet sourceもstride-2 convolutionを2段使います。
Gaussian sampling `x0 = mu + exp(0.5*logvar)*epsilon`、fixed std、learned
logvar、variance loss、`mu_tanh_scale`は維持しています。

教師ありsource lossは排他的に選びます。

```yaml
model:
  state_downsample_factor: 4
source:
  supervision:
    type: align          # align | cross_entropy | none
    weight: 0.15
```

`align`はstate `mu`をfull resolutionへbilinear拡大して既存のnormalized
alignmentを計算します。GTのfull one-hotは生成せず、normalized `mu`の二乗和と
integer GT classに対する`gather`から数学的に等価な値を求めます。
`cross_entropy`は`mu`をraw logitsとしてfull resolution
へ拡大し、softmaxせずCEへ渡します。旧`use_loss_align/align_weight`だけを持つ
configもload時に新形式へ変換されます。

### GT、loss、推論

`target_full`とnearest-resizeした`target_state`を明示的に分離します。
`one_hot_state`はFlow path専用、`valid_mask_full`はprimary/source supervision、
`valid_mask_state`はconsistency専用です。ADE20Kでは全maskが`target != 0`、
primary/source CEは`ignore_index=0`です。151 stateおよび評価class 1..150は
変更していません。Cityscapesは20 state、void index 19、評価19 classを維持します。

training Dataset/DataLoaderの返り値は両datasetとも`(image, target_full)`だけです。
CPUでもGPUでも`[B,C,H,W]`のfull-resolution target one-hotは作りません。
GPUへinteger targetを転送した後、`prepare_state_targets()`がinteger GTをnearestで
state解像度へ縮小し、その`target_state`だけをone-hot化します。ADE20K 512×512では
`one_hot_state`は`[B,151,128,128]`であり、旧`[B,151,512,512]`の1/16要素数です。

推論はstateのC-channel continuous terminal outputをpadded入力解像度へbilinear
resizeし、padding除去、original GT解像度へのbilinear resize、最後にargmaxの順です。
ADE20Kのaspect ratio保持、幅2048以下・高さ512以下、size divisor、original-resolution
評価、nanmeanの既存仕様を維持します。

## セットアップと主要config

```bash
cd /home/igarashi_25/DFM
uv sync --extra full
```

| dataset | diagonal | joint PSD |
|---|---|---|
| Cityscapes | `configs/diagonal_cityscapes.yaml` | `configs/joint_psd_cityscapes.yaml` |
| ADE20K | `configs/diagonal_ade20k.yaml` | `configs/joint_psd_ade20k.yaml` |

学習例:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py \
  --config configs/diagonal_ade20k.yaml

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  src/train_joint.py --config configs/joint_psd_ade20k.yaml
```

長時間学習を始める前に、`training.max_optimizer_steps`、dataset path、global batch、
`checkpoint.resume`、wandb設定を確認してください。

## 全体構成

経路とmean-denoiser Flow Mapは次のとおりです。

\[
x_t=(1-t)x_0+t x_1,\qquad
X^\theta_{s,t}(x_s)
=x_s+\frac{t-s}{1-s}\left(\psi^\theta_{s,t}(x_s,I)-x_s\right).
\]

`x_1`はvoidを含む20-class one-hotです。`x_0`は`source.prior_type`で
`gaussian`、`dirichlet`、`image_gaussian`から選びます。
`flow.time_eps`はFlow Mapの分母のゼロ除算防止にだけ使います。

学習方式は2つです。

| 方式 | entrypoint | 初期値 | iterationの損失 |
|---|---|---|---|
| Stage 1 → Stage 2 | `src/train.py` | Stage 2はStage 1 checkpoint | Stage 1: 対角CEのみ、Stage 2: 対角CE + 1種類の整合性損失 + source |
| joint | `src/train_joint.py` | endpointをランダム初期化 | epoch 1から対角CE + 1種類の整合性損失 + source |

総損失はStage 2とjointで共通です。

```text
loss_total =
    primary.weight * loss_diagonal
  + consistency.weight * consistency.max_weight * schedule * loss_consistency
  + source.var_weight * loss_source_var
  + source.supervision.weight * loss_source_supervision
```

`schedule`は`start_epoch`と`warmup_epochs`によるlinear warm-upです。

### Stage 1

`experiment.stage: diagonal_pretrain`を使います。モデル入力は
`(x_t, image, s=t, t=t)`で、endpointの学習目的は対角CEだけです。既存の
image-conditioned sourceを使う場合のsource regularizerは維持しますが、
PSD/CSD/ECLD/ESDのdispatchにも`torch.func.jvp`にも入りません。この回帰条件は
unit testでも禁止関数に置き換えて検証しています。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage1_diagonal_cityscapes.yaml
```

### Stage 2

`experiment.stage: consistency_distillation`を使い、`checkpoint.init_from`に
Stage 1 checkpointを指定します。旧`esd_distillation`はESD checkpointの
後方互換aliasとしてload/resume時も受理します。整合性損失は1 runにつき1種類で、
`loss.consistency.type`を`psd`、`csd`、`ecld`、`esd`から選びます。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage2_ecld_cityscapes.yaml
```

### Joint training（対角事前学習なし）

`src/train_joint.py`と`experiment.stage: joint_training`を使います。
`checkpoint.init_from`は禁止され、`resume`だけが許可されます。各iterationで
source priorを生成し、対角CE用の時刻と整合性用の時刻を別々にsampleして、
両損失を同時にbackwardします。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train_joint.py \
  --config configs/joint_ecld_cityscapes.yaml
```

## 整合性損失

共通入口は`compute_consistency_loss(...) -> ConsistencyResult`です。teacherは
stop-gradient、studentは勾配を保持し、損失固有の統計名は衝突しません。

### PSD

`s < u < t`の3時刻をsampleします。`s→u→t`のcomposed Flow Map teacherと
`s→t`のdirect Flow Map studentを比較し、teacher probabilityを再正規化して
detachします。PSDはJVPを使いません。そのため設定は必ず次の形です。

```yaml
precision:
  jvp_dtype: null
  numerical_dtype: fp32
```

PSDへ`bf16`/`fp32` JVPを指定するとconfig validation errorになります。

### CSD

`s < t`をsampleし、Flow Mapの時刻方向JVPと時刻`t`のinstantaneous diagonal
teacherからFP32 residualを作り、二乗ノルムを最小化します。teacher全体は
detachされます。`loss_csd`、residual norm、JVP平均/最大絶対値、dtype codeを
記録します。

### ECLD

時刻方向logits JVPから、完全Jacobianを作らずexact softmax JVP

\[
\dot p=p\odot\left(\dot z-\langle p,\dot z\rangle\mathbf 1\right)
\]

を計算します。transport後のendpoint teacherに対するCEと、`gamma(s,t)^2`で
重み付けしたtemporal derivative lossを`ec_weight`、`td_weight`で合成します。
`time_weighting`は`none`または`inverse_square`です。

### ESD

本実装のESDは、Discrete Flow Maps論文で導出された
**stabilized logit-space ESD**を基礎とします。logit-space teacher、
joint JVP、softmax gauge centering、終端付近で悪条件となる係数を直接計算しない
安定形、およびteacherからstudentへのforward KLはDFM論文由来です。
stabilized ESD自体を本リポジトリ独自の手法として導入したものではありません。

既存DFM式を保持しています。対角drift

\[
b_s=\frac{\psi_{s,s}(x_s)-x_s}{1-s}
\]

に沿うjoint JVP over `(x_s, s)`を計算し、softmax gaugeを中心化します。

\[
\delta=D_s z_{s,t}-\langle\psi_{s,t},D_s z_{s,t}\rangle\mathbf 1,
\]

```python
log_arg_raw = (
    one_minus_t[:, None, None, None]
    - (one_minus_s * delta_time)[:, None, None, None] * delta
)
```

teacherは`z_ss - log(clamped_log_arg)`から作り、損失方向は
`KL(teacher || student)`です。invalid class/pixel/sample率、nonfinite率、
clamp率、valid率、時刻bucket、teacher entropy、adaptive KL weight、
skip有無を記録します。`clamp`、`mask_pixel`、`skip_batch`を選べます。
全画素invalidでもstudent graphを保持するzero lossを返し、対角CEのbackwardを
継続します。NaN/Infを無条件に0へ置換して問題を隠す実装ではありません。

本リポジトリは論文由来のstabilized logit-space ESDに加え、
implementation-level numerical safeguardsとして、invalid log argument検出、
`log_eps` clamp、`mask_pixel`、`skip_batch`、clamp/invalid率の診断、
optional adaptive KL weighting、そのmean normalizationと最大値clamp、
consistency weight warm-up、bf16/FP32 JVP切り替え、JVP後のFP32数値計算を
提供します。論文由来の安定化と、これら追加安全策は区別してください。

ESDの由来と実装形は実験条件metadataにも保存されます。

```yaml
loss:
  consistency:
    esd:
      formulation: stabilized_logit_space
      source: discrete_flow_maps
      additional_numerical_safeguards: true
```

`additional_numerical_safeguards: true`は、この実装が追加安全策を備えることを
示すmetadataです。実際に有効な条件は、次の個別設定から判断します。

- `invalid_teacher.strategy`、`log_eps`、`skip_batch_threshold`
- `adaptive_kl.enabled`、`c`、`r`、`normalize_mean`、`max_weight`
- `consistency.weight`、`consistency.warmup_epochs`
- `precision.jvp_dtype`、`precision.numerical_dtype`

実験結果では`DFM-ESD`と表記し、必要に応じて
`DFM-ESD (mask_pixel, adaptive KL, bf16 JVP)`または
`DFM-ESD + numerical safeguards`のように詳細条件を併記してください。
`ESD (stabilized)`だけでは、DFM論文由来の安定化と本実装の追加安全策の区別が
曖昧になるため推奨しません。

## bf16 JVPとFP32 JVP

CSD/ECLD/ESDはYAMLで切り替えます。既定はbf16です。

```yaml
runtime:
  amp: true
  amp_dtype: bf16
loss:
  consistency:
    precision:
      jvp_dtype: bf16  # bf16 | fp32
      numerical_dtype: fp32
      debug_assertions: false
```

`runtime.amp: false`とbf16 JVPの組合せ、`numerical_dtype: bf16`、CUDAでbf16
非対応の環境は明示的なエラーです。比較時は既存キーをoverrideできます。

```bash
uv run python src/train.py --config configs/debug_ddp_stage2_ecld.yaml \
  --set loss.consistency.precision.jvp_dtype=fp32
```

bf16 pathではモデルforward、JVP内forward、JVP出力、teacher用forwardをbf16に
し、直後にFP32へ戻します。softmax、log-softmax、exact softmax JVP、Flow Map
係数、CSD residual、ECLD CE/TD、ESD delta/log/teacher/KL/adaptive weight、
全diagnosticsと最終lossはFP32です。FP32へ戻すのは、確率の正規化、logの境界、
小さな差分、KLをbf16の狭い仮数で評価しないためです。

`debug_assertions: true`ではJVP前後、student/teacher probability、lossのdtypeを
assertします。ログの`*_jvp_dtype_code`はFP32=`0`、bf16=`1`です。実GPU debug
ではCSD/ECLD/ESDの全runで`1`を確認しました。

## DDP設計

`distributed.enabled: auto`は`WORLD_SIZE > 1`でDDPを有効にし、通常の
`python src/train.py`ではsingle-processへfallbackします。本学習はNCCL、
CPU unit testはGlooです。

```yaml
distributed:
  enabled: auto
  backend: nccl
  init_method: env://
  find_unused_parameters: false
  broadcast_buffers: false
  gradient_as_bucket_view: true
```

学習用の`DDPCompatibleTrainingModel`はendpoint modelとsource modelを1つの
composite moduleとして所有します。Stage 1、Stage 2、jointの完全なforward
graphとJVPを、このcompositeへの1回のDDP `forward`内で構築します。学習損失を
作るために`ddp_model.module.forward_logits(...)`を外側から呼びません。
validation、inference、checkpoint保存時だけunwrapします。このためtrainable
sourceのgradientもendpointと同じDDP reducerで同期されます。frozen sourceは
gradientを持ちません。

学習DataLoaderは`DistributedSampler`を使い、各epochで`set_epoch(epoch)`を
呼びます。gradient accumulation中のoptimizer stepを行わないmicro stepは
`no_sync()`を使い、epoch末または`max_iterations`末の端数もstepします。
schedulerはCFMと同じepoch時間軸です。warm-upは`LinearLR`、その後は
`CosineAnnealingLR`を使い、各epochのsummary記録後に1回だけstepします。
同一epoch内のLRは一定で、source parameter groupにもmodel groupと同じ倍率を
適用します。`training.log_interval`は旧iterationログ用のdeprecated設定として
読み込み互換性のために残しますが、学習時には使用しません。

### Global batch size

`training.batch_size`は常にglobal batch sizeです。

```text
local_batch_size = global_batch_size // world_size
effective_global_batch_size = global_batch_size * grad_accum_steps
```

割り切れない場合は開始前にエラーです。2 GPUでglobal batch 4なら各rankの
local batchは2です。world size、rank、local rank、global/local/effective
batch、accumulationを開始ログへ記録します。

### 2 GPU実行

```bash
cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/stage2_ecld_cityscapes.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_ecld_cityscapes.yaml
```

対応スクリプトは`scripts/train_stage2_{psd,csd,ecld,esd}_ddp.sh`と
`scripts/train_joint_{psd,csd,ecld,esd}_ddp.sh`です。

## Checkpoint、ログ、評価

rank 0だけがwandbを初期化・記録し、通常ログ、`metrics.jsonl`、
`config_resolved.yaml`、checkpoint、可視化、evaluation JSONを書きます。
学習統計は各rankでepoch中にdetach済みtensorとして蓄積し、epoch終了時だけ
mean/min/maxをまとめてreduceします。iteration recordは保存しません。
W&B学習metricと`train_log.txt`も1 epochにつき1回です。GPU peakはrank別リスト、
rank平均、rank最大を保存します。

checkpoint保存の前後にbarrierを置き、state dictには`module.` prefixを付けません。
保存内容はstage、epoch、global step、model、source model、optimizer、
scheduler、scheduler step unit/version、scaler、resolved config、model signature、
metrics、world size、global/local batchです。resumeは完全復元し、world size変更は許容します。
global batch変更時は警告します。旧`module.`付きstate dictもload時に除去します。
旧optimizer-step scheduler checkpointはLRを誤復元しないよう明示的に拒否します。

`init_from`はStage 1のmodel/source重みだけを読み、optimizer等を初期化します。
joint checkpointは`stage: joint_training`で、joint同士だけresume可能です。
Stage 1/2 checkpointをjoint resumeへ渡すとstage mismatch errorになります。

YAMLは`extends: base.yaml`で同じディレクトリの設定を継承でき、派生設定は
上書き部分だけを保持します。load後の完全な設定は各runの
`config_resolved.yaml`へ保存されるため、実行条件は常に再現できます。
本学習用のStage 2およびjoint configは、各実験条件を単独で確認できるよう、
他configを継承しない自己完結型YAMLとして定義しています。

validation/evaluationはpaddingしない`DistributedEvalSampler`を使うため、全画像を
ちょうど1回だけ評価します。各rankの20×20 confusion matrixをSUM all-reduce
してからglobal mIoU、pixel accuracy、mean class accuracy、class-wise IoUを
計算します。GT class 19だけを除外し、prediction class 19は誤予測として残します。

DDP evaluationも可能です。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --standalone --nproc_per_node=2 src/evaluate.py \
  --config configs/stage2_ecld_cityscapes.yaml \
  --checkpoint /path/to/best.pt
```

## Debugと実測結果

全debug設定は48×96、1 epoch、2 iterations、global batch 4、DDP local batch 2、
bf16 AMP、wandb disabledです。Stage 1を作った後、次ですべて実行できます。

```bash
scripts/debug_all_ddp.sh
```

2026-07-26、NVIDIA RTX 6000 Ada 2枚、PyTorchのpeak allocated memory実測です。
全runでloss/gradientはfinite、optimizer step成功、checksum差0、metricsの
iteration行は2行、checkpointはrank 0の1組だけでした。時刻はwarm-up/初回JVP
オーバーヘッドを除くiteration 2です。

| 方式 | loss | total loss | consistency loss | grad norm | iter 2 (s) | max peak/rank (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2 | PSD | 1.80965 | 2.94146 | 0.57854 | 0.0594 | 89.90 |
| Stage 2 | CSD bf16 | 1.51629 | 0.00207 | 0.57855 | 0.1275 | 163.98 |
| Stage 2 | ECLD bf16 | 2.69967 | 11.83520 | 0.57946 | 0.1215 | 163.91 |
| Stage 2 | ESD bf16 | 1.51719 | 0.01101 | 0.57843 | 0.1139 | 173.70 |
| joint | PSD | 1.81143 | 2.94687 | 0.61167 | 0.0579 | 90.60 |
| joint | CSD bf16 | 1.51720 | 0.00443 | 0.61142 | 0.1433 | 164.69 |
| joint | ECLD bf16 | 2.70712 | 11.90321 | 0.61270 | 0.1394 | 164.62 |
| joint | ESD bf16 | 1.51707 | 0.00333 | 0.61172 | 0.1097 | 174.40 |

### メモリ比較

Stage 2の同じdebug model、global batch 4で比較しました。single GPUはlocal
batch 4、DDPは各rank local batch 2です。

| loss / 条件 | JVP code | loss (iter 2) | grad norm | iter 2 (s) | max peak (MiB) |
|---|---:|---:|---:|---:|---:|
| ECLD single GPU FP32 | 0 | 2.69718 | 0.59394 | 0.1166 | 419.50 |
| ECLD DDP FP32 | 0 | 2.69962 | 0.57946 | 0.1326 | 264.85 |
| ECLD DDP bf16 | 1 | 2.69967 | 0.57946 | 0.1215 | 163.91 |
| ESD single GPU FP32 | 0 | 1.51595 | 0.59401 | 0.0820 | 355.93 |
| ESD DDP FP32 | 0 | 1.51720 | 0.57843 | 0.1115 | 189.61 |
| ESD DDP bf16 | 1 | 1.51719 | 0.57843 | 0.1139 | 173.70 |

local batch分割により、DDP FP32のrank最大はsingle FP32比でECLD 36.9%、
ESD 46.7%減りました。さらにbf16 JVPはDDP FP32比でECLD 38.1%、ESD 8.4%
減りました。DDP自体が「1 GPU内の1サンプル当たりメモリ」を減らすわけではなく、
同じglobal batchを複数GPUのlocal batchへ分割した結果、各GPUのbatch由来メモリ
が減ります。時間は2 iterationだけのdebug測定であり、性能benchmarkでは
ありません。

## テスト

```bash
uv run pytest -q
```

config strictness、4損失と時刻順序、PSD JVP禁止、bf16/FP32の実dtypeとFP32
post-processing、softmax JVP、teacher detach、ESD invalid/adaptive処理、
Stage回帰、joint、checkpoint、void評価、CPU/Gloo 2-process reduction、
no_sync、endpoint/source gradient同期、frozen source、rank 0保存、
non-padding samplerを検証します。

## 既知の制限

- endpoint modelは現在UNetのみです。sourceはSegFormerと軽量UNetです。
- 主対象は単一ノードNCCLです。multi-nodeの性能・障害復旧は未検証です。
- ESD joint JVPとCSD/ECLD JVPは通常の対角forwardよりメモリを使います。
- `source.pretrained: true`の初回はHugging Face weightまたはcacheが必要です。
- bf16の可否はCUDA deviceで実行開始時に検証します。FP16 JVPは未対応です。
- debug runは配線、dtype、finite backward、DDP同期確認用で、精度評価では
  ありません。
