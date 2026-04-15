import os
import time
import gc
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformers
import numpy as np
import torch.nn.functional as F

from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from flatquant.quant_utils import set_quantizer_state


def load_svd_loss_state(args, dev):
    if not getattr(args, "svd_loss", False):
        return None
    if args.svd_file is None:
        raise ValueError("--svd_loss is enabled but --svd_file is not provided.")

    svd_results = np.load(args.svd_file)
    s = torch.from_numpy(svd_results['s']).to(torch.float32)
    Vh = torch.from_numpy(svd_results['Vh']).to(torch.float32)
    V = Vh.transpose(0, 1).contiguous()

    if args.svd_weight_mode == "sigma2":
        w = s.pow(2)
    elif args.svd_weight_mode == "sigma2_norm":
        w = s.pow(2)
        w = w / w.mean()
    elif args.svd_weight_mode == "sigma2_norm_clip_low":
        w = s.pow(2)
        w = w / w.mean()
        s_mean = s.mean()
        w = torch.where(s > s_mean, w, torch.ones_like(w))
    else:
        raise NotImplementedError(f"Unsupported svd_weight_mode: {args.svd_weight_mode}")

    return {
        "V": V.to(dev),
        "w": w.to(dev),
        "s2": s.pow(2).to(dev),
    }


def get_layer_svd_alpha(args, layer_idx, num_layers):
    if not getattr(args, "svd_loss", False):
        return 0.0

    alpha_end = float(args.svd_loss_alpha)
    if args.svd_alpha_schedule == "constant" or num_layers <= 1:
        return alpha_end
    if args.svd_alpha_schedule == "linear":
        alpha_start = float(args.svd_alpha_start)
        return alpha_start + (alpha_end - alpha_start) * (layer_idx / (num_layers - 1))
    raise NotImplementedError(f"Unsupported svd_alpha_schedule: {args.svd_alpha_schedule}")


def compute_svd_weighted_loss(fp_out, quant_out, svd_state):
    if svd_state is None:
        raise ValueError("svd_state is required for SVD-weighted loss computation.")

    err = fp_out - quant_out
    err_proj = torch.matmul(err, svd_state["V"].to(err))
    weighted_err = err_proj.pow(2) * svd_state["w"].to(err)
    return weighted_err.mean()


def compute_recon_loss(fp_out, quant_out, svd_state=None, svd_alpha=1.0):
    plain_mse = compute_plain_mse(fp_out, quant_out)
    if svd_state is None or svd_alpha <= 0:
        return plain_mse, plain_mse

    svd_weighted_loss = compute_svd_weighted_loss(fp_out, quant_out, svd_state)
    loss = (1 - svd_alpha) * plain_mse + svd_alpha * svd_weighted_loss
    return loss, plain_mse


def compute_plain_mse(fp_out, quant_out):
    return F.mse_loss(fp_out, quant_out)


def compute_head_proxy_mse(fp_out, quant_out, svd_like_state):
    err = fp_out - quant_out
    err_proj = torch.matmul(err, svd_like_state["V"].to(err))
    weighted_err = err_proj.pow(2) * svd_like_state["s2"].to(err)
    return weighted_err.mean()

def cali_flat_quant(args, model, dataloader, dev, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = inps   # take output of fp model as input
    fp_outs = torch.zeros_like(inps)   # take output of fp model as input

    svd_state = load_svd_loss_state(args, dev)
    if svd_state is not None:
        logger.info(f"Using SVD-weighted reconstruction loss from: {args.svd_file}")
        logger.info(f"SVD weight mode: {args.svd_weight_mode}")
        logger.info(
            "SVD loss mix: schedule=%s alpha_start=%.6f alpha_end=%.6f"
            % (args.svd_alpha_schedule, args.svd_alpha_start, args.svd_loss_alpha)
        )
        w = svd_state["w"]
        logger.info(
            "SVD weight dynamic range: min=%.6f max=%.6f ratio=%.6f"
            % (w.min().item(), w.max().item(), (w.max() / w.min()).item())
        )
    if svd_state is None:
        with torch.no_grad():
            lm_head_weight = model.lm_head.weight.detach().to(torch.float32)
            _, s_head, Vh_head = torch.linalg.svd(lm_head_weight, full_matrices=False)
            head_proxy_state = {
                "V": Vh_head.transpose(0, 1).contiguous().to(dev),
                "s2": s_head.pow(2).to(dev),
            }
            del lm_head_weight, s_head, Vh_head
            torch.cuda.empty_cache()
    else:
        head_proxy_state = {
            "V": svd_state["V"],
            "s2": svd_state["s2"],
        }
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
        layer_svd_alpha = get_layer_svd_alpha(args, i, num_train_layer)
        logger.info(f"Layer {i} SVD alpha: {layer_svd_alpha:.6f}")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False
        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init == "one_style":
            pass
        else:
            raise NotImplementedError

        layer = layer.to(dev)
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // args.cali_bsz), eta_min=args.flat_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        for epoch in range(args.epochs):
            optimized_loss_sum = 0
            plain_mse_sum = 0
            head_mse_sum = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    fp_batch = fp_outs[index:index+args.cali_bsz,]
                    quant_out = layer(fp_inps[index:index+args.cali_bsz,], attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                    loss, plain_mse = compute_recon_loss(
                        fp_batch.float(), quant_out.float(), svd_state=svd_state, svd_alpha=layer_svd_alpha
                    )
                    head_proxy_mse = compute_head_proxy_mse(fp_batch.float(), quant_out.float(), head_proxy_state)
                    optimized_loss_sum += loss.detach().cpu()
                    plain_mse_sum += plain_mse.detach().cpu()
                    head_mse_sum += head_proxy_mse.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, "
                f"optimized_loss: {optimized_loss_sum:.8f}, plain_mse: {plain_mse_sum:.8f}, head_proxy_mse: {head_mse_sum:.8f}"
            )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model
