> 历史对齐记录，不能据此认定队友当前 run 的配置。原文来自团队仓库 yf/docs/B1_ALIGNMENT.md。

# B1 设置对齐：yf vs. 主线代码

目的：弄清楚为什么 yf 的实验里 B1 效果明显，而主线的 `train_own_b1_reps.py` 里加 inverse loss 前后差别很小。下面按 yf 的 `nav/train_nav.py` 和主线的 `training/train_own_b1_reps.py`、`cluster/b1_long_train.sbatch` 逐项对比。

## 1. 一致的部分

B1 的实现本身基本一致：

- inverse head：MLP `Linear(2·192→512)-GELU-Linear(512→512)-GELU-Linear(512→A)`
- 输入是 encoder + projector（BatchNorm）之后的 `emb`，**不 detach**，梯度会传回 encoder
- 覆盖 clip 内所有相邻帧对，`a_t` 是从观测 t 到观测 t+1 的 block 位移
- λ = 1
- 连续表示：对 z-score 之后的动作算 MSE；disc 和 dbin：分组交叉熵，dbin 的分箱数是 3/5/7，用 train 集分位数拟合
- 其余部分都是 LeWM 原版：ViT-tiny，history 3，num_preds 1，SIGReg 0.09，AdamW lr 5e-5，LinearWarmupCosineAnnealing，grad clip 1.0

## 2. 不一致的部分

| | yf（`nav/train_nav.py`） | 主线（`cluster/b1_long_train.sbatch`） |
| --- | --- | --- |
| **训练量** | 10 个完整 epoch，batch 128，约 1 万步，相当于约 130 万个样本（按 val 的 clip 数推算的估计值） | 30 epoch × `limit_train_batches=100`，batch 32，**共 3000 步，约 9.6 万个样本** |
| batch size | 128 | 32 |
| 数据 | NavDP 原始轨迹，用 `nav/convert_navdp.py` 转换；每帧存一行，按 `frameskip=8` 取观测，`clip_stride=4` | MP3D `nav_waypoint8` HDF5，每 8 帧预先抽一行，`frameskip=1` |
| 场景划分 | [`yf/data/splits.json`](../data/splits.json)：场景名排序后 `random.Random(0).shuffle`，54/7/7 | `scene_split_54_7_7.json`，数量相同，**但不是同一批场景**：主线 Habitat 验证用的 `JF19kD82Mey`、`jtcxE69GiFV` 在 yf 这份里属于 train |
| 动作坐标 | x 朝前、y 朝左，dθ>0 为左转，有单元测试 | 转换到 [前, 左, yaw]，注释里写的是只做了抽样检查 |
| 动作归一化 | 用 train 全部行的 block 位移计算每维的 mean/std，具体数值见 [`yf/data/action_norm.json`](../data/action_norm.json)（例如 fs8 下 dx 的 mean 0.294 m、std 0.042 m） | 用 `get_column_normalizer` 在 train 上拟合，口径大致相同，但具体数值还没有比对 |
| 精度 | bf16-mixed | bf16 |
| 评估 | `eval_latent.py`：val 上最多 2 万个 clip；看相对 copy-last 的比值、动作打乱增益、动作辨别 top-1、多步 rollout | 训练时只用 20 个 val batch；另外有自己的评估脚本 |

## 3. 判断

1. **最可能的原因：训练量差了大约 13 倍。** B1 起作用的方式是逐渐让 latent 编码自运动、让 predictor 依赖动作。在 3000 步、batch 32 的规模下，这个效果可能还没出现。
2. **指标。** 加了 B1 之后，**绝对 MSE 会变高**：fs8 下从 0.052 变成 0.082，因为 copy-last 的误差也从 0.058 涨到 0.122。所以只看预测 loss 曲线，看不出 B1 的提升，要看下面三个：

   | 指标（fs8，xyt，seed 3072） | 不加 B1 | +B1（λ=1） |
   | --- | --- | --- |
   | 单步 模型/copy-last | 0.895 | 0.671 |
   | 1.2 m 处 模型/copy-last | 1.032 | 0.754 |
   | 动作打乱后误差增加 | +6.1% | +80.9% |
   | 动作辨别 top-1 | 0.112 | 0.299 |

3. **frameskip 也有影响。** 在 fs4 下，B1 的单步比值反而变差（0.742 → 0.783），好处只体现在多步 rollout 上。

## 4. 建议的对齐顺序

1. **训练量**：去掉 `limit_train_batches`，batch 128，跑 10 个完整 epoch；或者两边约定一个相同的总步数。
2. **场景划分**：比较 `yf/data/splits.json` 和 `scene_split_54_7_7.json`，统一用其中一份。planning 和 Habitat 验证尤其要注意：用 yf 的模型时，不能在 `JF19kD82Mey`、`jtcxE69GiFV` 上测，因为这两个场景在 yf 的 train 集里。
3. **评估**：统一看相对 copy-last 的比值、动作打乱增益、top-1，可以直接用 `nav/eval_latent.py`。
4. 如果前三项对齐后结果还对不上，再比对数据转换：抽几条同一场景、同一轨迹的样本，对比动作的数值和归一化参数。
