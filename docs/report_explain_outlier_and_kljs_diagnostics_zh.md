# 两次 Outlier / KL-JS 诊断实验说明

## 1. 为什么要做这两次实验

老师提到的核心意思是：量化方法不能只看最终 PPL 或 zero-shot 分数。一个方法如果真的让模型更好量化，通常应该能从 tensor 分布上看到变化。

LLM 量化困难的一个重要原因是 outlier。

简单说，4-bit 量化只有很少的离散刻度。如果一个 tensor 里大多数值都很小，但有少量特别大的 outlier，那么量化范围会被这些特别大的值撑开，导致大多数普通值被分配到很粗的刻度上，量化误差就会变大。

所以很多 PTQ 方法本质上都在做类似的事情：

- 找到 weight 或 activation 分布中不适合量化的部分。
- 通过 rotation、scaling、clipping、low-rank 分离等方式处理 outlier。
- 让变换后的分布更平滑、更少极端值、更接近量化器容易处理的形态。

老师说的“看 histogram”“看 KL”“看信噪比”“多尺度看”，本质上都是为了回答同一个问题：

> 我们的方法到底有没有真的把难量化的分布变成更好量化的分布？

我们之前已经看过 PPL、zero-shot、`head_proxy_mse`。这些指标回答的是最终效果或 head-sensitive error geometry。但它们不能直接回答“weight / activation 分布到底变没变好”。因此我们做了这两次诊断实验。

## 2. 第一次实验做了什么

第一次实验是：

- `Qwen2.5-3B Base Outlier 分布诊断`
- 报告：`docs/report_outlier_distribution_qwen25_base_20260528_zh.md`
- 脚本：`diagnostics/qwen25_outlier_distribution.py`

### 2.1 它比较了什么

它比较三种模型状态：

| 状态 | 含义 |
| --- | --- |
| `fp` | 原始浮点模型，还没有 FlatQuant 变换 |
| `baseline` | established FlatQuant baseline |
| `svd_mix_linear` | 当前 A+B SVD-FlatQuant 方法 |

这三个状态的关系是：

```text
FP 原始分布
  -> baseline FlatQuant 之后的分布
  -> SVD A+B 之后的分布
```

我们想知道：

1. baseline 是否真的把 FP 的 outlier 变好了。
2. SVD A+B 是否在 baseline 基础上进一步把 outlier 变好了。

### 2.2 它看了哪些 tensor

它看每层主要 linear module：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

这些就是 transformer block 里主要的 weight 和 activation 量化位置。

### 2.3 它用了什么指标

第一次实验用了几类指标。

第一类是 outlier ratio，例如：

- `abs_max_over_p99`
- `p999_over_p50`
- `token_score_p99`
- `channel_max_over_median`

这些指标大致表示“极端值相比普通值有多夸张”。如果这些值变小，说明 outlier 没那么严重。

第二类是 fake quantization error，例如：

- `w4_sym_per_out_mse`
- `a4_sym_token_mse`

它们直接模拟 W4 或 A4 量化后的误差。越小越好。

第三类是 SQNR：

- `w4_sym_per_out_sqnr_db`
- `a4_sym_token_sqnr_db`

SQNR 可以理解成“信号相比量化噪声有多强”。越大越好。老师说的“看信噪比”对应的就是这类指标。

第四类是图：

- heatmap
- histogram

heatmap 用来看全层趋势，histogram 用来看具体分布形状。

### 2.4 第一次实验得到了什么结论

第一次实验最重要的结论是：

> FlatQuant baseline 已经非常明显地改变了 FP 的 outlier-heavy 分布。

例如 activation 上：

| 指标 | FP | Baseline |
| --- | ---: | ---: |
| `token_score_p99` mean | `119.0528` | `5.8646` |
| `a4_sym_token_sqnr_db` mean | `6.4133` | `16.3468` |

这说明原始 FP activation 里有很严重的 token outlier，而 baseline FlatQuant 把它压下来了，同时 A4 量化信噪比大幅提升。

但 SVD A+B 相比 baseline 没有明显进一步改善普通分布：

| 指标 | Baseline | SVD A+B |
| --- | ---: | ---: |
| activation `token_score_p99` mean | `5.8646` | `5.8688` |
| activation `a4_sym_token_sqnr_db` mean | `16.3468` | `16.3404` |
| weight `abs_max_over_p99` mean | `2.6596` | `2.5746` |
| weight `w4_sym_per_out_sqnr_db` mean | `19.4555` | `19.4392` |

所以第一次实验回答了：

> SVD A+B 并没有像 baseline FlatQuant 那样带来明显的额外 distribution flattening 效果。

## 3. 第二次实验做了什么

第二次实验是：

- `Qwen2.5-3B Base KL/JS 分布差异与 Head-Proxy 对齐诊断`
- 报告：`docs/report_kl_js_head_alignment_qwen25_base_20260529_zh.md`
- 脚本：`diagnostics/qwen25_kl_js_head_alignment.py`

它是在第一次实验基础上继续做两件事。

第一件事：用 KL / JS divergence 量化分布到底差多少。

第二件事：把 `head_proxy_mse` 和 outlier / SQNR 指标对齐，看看 SVD 改善的“语义重要方向”是否也带来普通分布改善。

## 4. 什么是 KL / JS，为什么要算它

老师说“像蒸馏一样算一个 KL 散度”，意思是不要只肉眼看 histogram，而是把两个 histogram 当成两个概率分布，计算它们之间的距离。

如果两个分布非常像，KL/JS 就接近 0。

如果两个分布差很多，KL/JS 就比较大。

本实验主要看 JS divergence，因为它比 raw KL 更稳定，也更容易解释。

我们比较了三组分布距离：

| Pair | 目的 |
| --- | --- |
| FP vs Baseline | baseline 到底有没有改变原始分布 |
| Baseline vs SVD A+B | SVD 是否在 baseline 上继续改变分布 |
| FP vs SVD A+B | SVD 最终状态和 FP 差多少 |

如果方法真的有明显额外分布效果，那么 `Baseline vs SVD A+B` 的 JS 应该不小。

但结果显示它非常小。

## 5. 第二次实验的 KL/JS 结论

Token outlier score 分布的 JS 结果：

| Pair | Mean JS |
| --- | ---: |
| FP vs Baseline | `0.687812` |
| Baseline vs SVD A+B | `0.000444` |
| FP vs SVD A+B | `0.687720` |

Activation 绝对值分布的 JS 结果：

| Pair | Mean JS |
| --- | ---: |
| FP vs Baseline | `0.132993` |
| Baseline vs SVD A+B | `0.000223` |
| FP vs SVD A+B | `0.134866` |

Weight 绝对值分布的 JS 结果：

| Pair | Mean JS |
| --- | ---: |
| FP vs Baseline | `0.136758` |
| Baseline vs SVD A+B | `0.000585` |
| FP vs SVD A+B | `0.137609` |

这些数字说明得很清楚：

- FP 到 baseline 的分布变化很大。
- FP 到 SVD A+B 的分布变化也很大。
- baseline 到 SVD A+B 的分布变化极小。

所以第二次实验用 KL/JS 的方式确认了第一次实验的结论：

> 分布变化主要来自 FlatQuant baseline，而不是 SVD A+B。

## 6. 什么是 `head_proxy_mse`，为什么要和 outlier 指标对齐

我们做 SVD loss 的初衷不是单纯降低普通 MSE，而是希望降低对最终 `lm_head` 更重要方向上的误差。

普通 MSE 看的是 hidden state 空间里所有方向平均误差。

`head_proxy_mse` 看的是经过 `lm_head` 谱结构加权之后的误差，也就是更接近最终 logits 敏感方向的误差。

所以如果 SVD 方法有效，它可能出现这种现象：

- 普通 MSE 没变好，甚至变差。
- 但 `head_proxy_mse` 变好。

这就是我们之前已经观察到的现象。

但老师又提醒我们看普通分布和量化友好性。因此第二次实验进一步问：

> 如果 `head_proxy_mse` 变好了，weight / activation outlier 和 SQNR 会不会也变好？

这就是“对齐诊断”的意义。

## 7. 第二次实验的 Head-Proxy 对齐结论

总模块数是 `252`。

| 指标 | 改善模块数 |
| --- | ---: |
| `head_proxy_mse` improved | `231 / 252` |
| `plain_mse` improved | `70 / 252` |
| activation `token_score_p99` improved | `106 / 252` |
| activation `a4_sym_token_sqnr_db` improved | `94 / 252` |
| weight `abs_max_over_p99` improved | `146 / 252` |
| weight `w4_sym_per_out_sqnr_db` improved | `66 / 252` |

这个结果非常关键。

它说明：

- SVD A+B 确实在绝大多数层上改善了 `head_proxy_mse`。
- 但普通 `plain_mse` 没有同步改善。
- activation token outlier 没有同步改善。
- A4 / W4 SQNR 也没有同步改善。

换句话说：

> SVD A+B 改善的是 head-sensitive error geometry，而不是普通量化分布。

## 8. 为什么这个结论重要

这个结论帮助我们解释之前看起来有点矛盾的结果。

之前我们看到：

- SVD A+B 的 `head_proxy_mse` 经常更好。
- 但 PPL 没有全面超过 baseline。
- zero-shot 也只是非常小幅变化。

现在分布诊断告诉我们原因可能是：

- SVD A+B 确实在某些“语义重要方向”上改变了误差。
- 但它没有让普通 activation / weight 分布更好量化。
- 它也没有稳定提高 A4/W4 SQNR。

所以它的收益不足以转化成稳定的最终性能提升。

## 9. 两次实验分别回答了什么

### 9.1 第一次实验回答的问题

第一次实验问：

> 我们的方法有没有把 outlier-heavy 的 tensor 分布变得更好量化？

答案是：

- baseline FlatQuant 有，非常明显。
- SVD A+B 相对 baseline 没有明显额外效果。

### 9.2 第二次实验回答的问题

第二次实验问：

> 用 KL/JS 这种分布距离来看，baseline 和 SVD A+B 的分布差异到底大不大？如果 SVD 改善了 head-sensitive loss，它是否也改善普通 outlier / SQNR？

答案是：

- baseline 和 SVD A+B 的普通分布差异非常小。
- SVD A+B 的 `head_proxy_mse` 改善很明显。
- 但这个改善没有同步体现在普通 outlier / SQNR 上。

## 10. 用一句话总结这两次实验

这两次实验共同说明：

> FlatQuant baseline 是主要的 outlier-removal / distribution-flattening 方法；当前 SVD A+B 主要是在改变 output-head-sensitive error geometry，而不是进一步把普通 weight / activation 分布变得更好量化。

## 11. 对后续研究的意义

如果我们要继续沿 SVD 方向推进，不能只继续调 SVD 权重或 alpha。因为现在的问题不是“不知道 SVD 有没有影响”，而是已经知道：

- 它对 `head_proxy_mse` 有影响。
- 但它对普通分布量化友好性影响不够。

因此下一步更合理的方向是设计一个同时满足两件事的方法：

1. 保留 SVD 对 head-sensitive error 的改善。
2. 不损害普通 MSE、activation outlier 和 A4/W4 SQNR。

可以考虑的方向包括：

- 在 mixed SVD loss 中加入 activation quantization noise proxy。
- 对 late layers 单独设计更温和的 SVD schedule。
- 把 `head_proxy_mse` 和普通 quantization SQNR 作为联合诊断指标，而不是只看其中一个。

## 12. 给不熟悉这些实验的直观解释

可以把模型里的 tensor 想象成一堆需要装进 4-bit 小盒子的数字。

如果数字分布很均匀，小盒子很好分配，量化误差小。

如果大多数数字很小，但有少数数字特别大，那么小盒子的范围会被特别大的数字撑开，普通数字就被挤在很少几个刻度里，误差变大。这就是 outlier 对量化不友好的原因。

第一次实验是在看：

> 这些数字经过我们的方法之后，是不是更适合装进 4-bit 小盒子？

第二次实验是在看：

> 如果两个数字分布看起来像不像，能不能用 KL/JS 给出一个数？以及 SVD 说它保留了“更重要的方向”，这件事和普通量化分布有没有关系？

最后得到的直观结论是：

- FlatQuant baseline 确实把数字整理得更适合 4-bit 量化。
- SVD A+B 不是继续整理这些数字的主要方法。
- SVD A+B 更像是在说“某些方向更重要，我优先照顾这些方向”。
- 但只照顾这些方向，还不足以让整个模型最终效果稳定变好。
