# Qwen2.5-3B Base 上 SVD 诊断实验报告

## 1. 实验目的

本轮实验的目标不是再次验证 FlatQuant baseline 能否跑通，而是回答一个更精确的问题：

> 在 `Qwen2.5-3B base` 上，引入 SVD-weighted loss 之后，与最终 `lm_head` 输出相关的误差到底有没有变小？

此前我们已经知道：

- 最佳稳定 baseline（`lwc + lac`）的最终 PPL 略优于当前 SVD 版本；
- 但旧日志里只有一个名为 `mse` 的字段，无法准确判断它到底代表普通 hidden-space 重构误差，还是 SVD 加权后的训练目标。

因此，本轮实验的核心工作是：

1. 明确原日志中 `mse` 的真实含义；
2. 加入更细粒度的诊断指标；
3. 重新比较 baseline 和 SVD 两次完整实验。

---

## 2. 原日志中的 `mse` 到底是什么

旧版本日志中，每层训练结束后会打印一个 `mse` 字段。

但这个量在两种实验里并不相同：

### 不带 SVD 时

日志中的 `mse` 就是普通重构 MSE：

$$
\mathcal{L}_{\text{plain}} = \operatorname{mean}\big((y_{fp} - y_{quant})^2\big)
$$

### 带 SVD 时

日志中的 `mse` 实际上是 SVD-weighted reconstruction loss：

$$
\mathcal{L}_{\text{svd}} = \operatorname{mean}\big((eV)^2 \odot w\big)
$$

其中：

$$
e = y_{fp} - y_{quant}
$$

因此，旧日志里的 `mse` 不能直接拿来回答：

- SVD 是否降低了普通 MSE；
- SVD 是否降低了与最终 `lm_head` 输出更相关的误差。

---

## 3. 数学定义

### 3.1 输出头的奇异值分解

设最终输出头权重为：

$$
W = \text{lm\_head.weight} = U \operatorname{diag}(s) V^T
$$

其中：

- $$U$$ 为左奇异向量；
- $$s$$ 为奇异值；
- $$V$$ 为右奇异向量。

设某层的浮点输出与量化输出之间的误差为：

$$
e = y_{fp} - y_{quant}
$$

### 3.2 普通重构 MSE

普通 hidden-space 重构误差定义为：

$$
\mathcal{L}_{\text{plain}} = \operatorname{mean}(e^2)
$$

它衡量的是：

- 当前层输出在普通欧式意义下离浮点输出有多远。

### 3.3 SVD-weighted loss

先将误差投影到输出头谱基底中：

$$
e_{proj} = eV
$$

然后按奇异值构造权重进行加权：

$$
\mathcal{L}_{\text{svd}} = \operatorname{mean}(e_{proj}^2 \odot w)
$$

本轮实验中使用：

$$
w = \frac{s^2}{\operatorname{mean}(s^2)}
$$

### 3.4 `head_proxy_mse` 的定义

如果显式地把误差通过输出头映射到 logits 空间，则有：

$$
\Delta z = eW^T
$$

显式 logits-space MSE 代价太高，因为 vocab 维度很大，直接计算会 OOM。

利用：

$$
W^T W = V \operatorname{diag}(s^2) V^T
$$

我们定义低显存的输出头代理误差：

$$
\mathcal{L}_{\text{head-proxy}} = \operatorname{mean}(e_{proj}^2 \odot s^2)
$$

这个量不是显式 logits 张量上的逐元素 MSE，但它对应同一个由 $$W^T W$$ 诱导的二次型，因此是一个正确的、与输出头敏感方向一致的低显存代理指标。

---

## 4. 本轮新增日志指标

为了解决旧日志的歧义，本轮我们在代码里新增并同时打印了三个量：

### 4.1 `optimized_loss`

表示实际被当前实验优化的目标：

- baseline：

$$
\text{optimized\_loss} = \mathcal{L}_{\text{plain}}
$$

- SVD：

$$
\text{optimized\_loss} = \mathcal{L}_{\text{svd}}
$$

### 4.2 `plain_mse`

表示普通 hidden-state 重构误差：

$$
\text{plain\_mse} = \mathcal{L}_{\text{plain}}
$$

### 4.3 `head_proxy_mse`

表示与最终输出头方向相关的代理误差：

$$
\text{head\_proxy\_mse} = \mathcal{L}_{\text{head-proxy}}
$$

---

## 5. 实验设置

### 模型

- `Qwen2.5-3B base`

### 公共量化配置

- `W4A4KV4`
- `nsamples=128`
- `epochs=15`
- `cali_bsz=4`
- `flat_lr=5e-3`
- `--cali_trans --add_diag`
- `--lwc --lac`
- `--deactive_amp --direct_inv`

### 实验 1：baseline 诊断实验

- GPU：`0`
- exp name：`qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0`
- 不启用 `--svd_loss`

### 实验 2：SVD 诊断实验

- GPU：`3`
- exp name：`qwen25_3b_base_w4a4kv4_lwc_lac_svd_full_headmse_gpu3`
- 启用：
  - `--svd_loss`
  - `--svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`
  - `--svd_weight_mode sigma2_norm`

---

## 6. 最终 PPL 对比

| 方法 | WikiText2 PPL | C4 PPL |
|---|---:|---:|
| Baseline (`lwc + lac`) | 8.8095 | 14.5661 |
| SVD (`lwc + lac + svd_loss`) | 8.8400 | 14.5785 |

### PPL 结论

- SVD 版本最终仍然略差于 baseline；
- 差距很小，但方向上并没有改善。

---

## 7. 关键层逐层对比

下面选取若干代表性层，展示三种指标的对比结果。

### 表 1：关键层指标对比

| Layer | 方法 | optimized_loss | plain_mse | head_proxy_mse |
|---|---|---:|---:|---:|
| 0 | Baseline | 0.17738493 | 0.17738493 | 16.23518562 |
| 0 | SVD | 0.17460741 | 0.17842296 | 15.79053974 |
| 2 | Baseline | 0.03333194 | 0.03333194 | 3.25202179 |
| 2 | SVD | 0.03506071 | 0.03405451 | 3.17069983 |
| 3 | Baseline | 0.04139658 | 0.04139658 | 3.77041078 |
| 3 | SVD | 0.04159188 | 0.04223246 | 3.76134276 |
| 20 | Baseline | 0.38674530 | 0.38674530 | 37.50878906 |
| 20 | SVD | 0.40922675 | 0.39270559 | 37.00822067 |
| 29 | Baseline | 1.39522707 | 1.39522707 | 127.90704346 |
| 29 | SVD | 1.40770745 | 1.41846216 | 127.30536652 |
| 30 | Baseline | 2.58813071 | 2.58813071 | 243.93521118 |
| 30 | SVD | 2.71171403 | 2.66353917 | 245.23257446 |
| 34 | Baseline | 8.62847614 | 8.62847614 | 897.73602295 |
| 34 | SVD | 9.61384487 | 9.07591629 | 869.42364502 |
| 35 | Baseline | 9.59827709 | 9.59827709 | 1187.19238281 |
| 35 | SVD | 12.44436741 | 10.40928078 | 1125.40075684 |

---

## 8. 如何理解这些对比

### 8.1 `optimized_loss` 的结论

`optimized_loss` 并没有在 SVD 版本里系统性变小。

尤其在后几层：

- Layer 34：SVD 更高
- Layer 35：SVD 显著更高

这说明：

- 当前 SVD-weighted objective 在优化上并不轻松；
- 至少从训练标量角度看，它没有呈现出“更容易被压低”的趋势。

但要注意：

- baseline 与 SVD 的 `optimized_loss` 不是同一个函数；
- 因此它不能被简单地当作一模一样的指标来比较。

### 8.2 `plain_mse` 的结论

SVD 版本在很多层上的 `plain_mse` 会略差于 baseline。

例如：

- Layer 20：SVD 更差
- Layer 34：SVD 更差
- Layer 35：SVD 明显更差

这说明：

- SVD 往往在牺牲普通 hidden-space 重构质量。

### 8.3 `head_proxy_mse` 的结论

这是本轮实验最重要的结果。

在不少层，尤其是一些后层，SVD 版本出现了：

- `plain_mse` 更差
- 但 `head_proxy_mse` 更小

典型例子：

- Layer 0：SVD 更好
- Layer 2：SVD 更好
- Layer 20：SVD 更好
- Layer 29：SVD 更好
- Layer 34：SVD 明显更好
- Layer 35：SVD 明显更好

这说明：

> SVD-weighted loss 不是在随机改变误差，而是确实在把误差往更有利于 `lm_head` 的方向移动。

---

## 9. 本轮实验的核心结论

这次实验最重要的结论不是“PPL 还是没提升”，而是我们终于明确了内部机制：

1. 当前 SVD 方法**确实会改变误差几何结构**；
2. 它往往会让普通 hidden-space 的 `plain_mse` 变差；
3. 但它在不少层，尤其最后几层，会让 `head_proxy_mse` 变小；
4. 也就是说，它确实在朝“更利于输出头”的方向优化；
5. 只是这种改善目前还没有强到足以带来更好的最终 PPL。

因此，当前最准确的判断是：

> 这条 SVD 路线不是完全无效，而是当前损失设计造成了一个 trade-off：它改善了输出头相关的代理误差，但同时损伤了普通重构质量，最终没有转化成更优的语言建模性能。

---

## 10. 对下一步工作的启发

如果继续沿 SVD 方向推进，重点就不是再验证“它有没有任何作用”，而是要研究：

1. 如何保留 `head_proxy_mse` 的改善；
2. 如何减少对 `plain_mse` 的破坏；
3. 如何把这种输出头方向上的改善真正转化为更好的 PPL。

当前最自然的下一步包括：

### 10.1 改成混合损失

例如：

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{plain}} + \beta \mathcal{L}_{\text{svd}}
$$

这样有望在保留输出头方向偏置的同时，不让普通重构质量损失过大。

### 10.2 弱化谱权重

当前使用的是 `sigma2_norm`，这可能过强。可以尝试更软的权重函数。

### 10.3 保留新增诊断日志

这次实验已经说明：

- 单独一个 `mse` 字段是远远不够的；
- 以后所有 SVD 相关实验都应该继续同时记录：
  - `optimized_loss`
  - `plain_mse`
  - `head_proxy_mse`

否则无法准确解释实验现象。
