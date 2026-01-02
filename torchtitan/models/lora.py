import os
import torch
from types import MethodType
from peft import LoraConfig, get_peft_model, TaskType

from timelyfreeze.core.config import TimelyFreezeConfig
from torchtitan.tools.logging import logger


def maybe_attach_lora(model_parts: list[torch.nn.Module], config: TimelyFreezeConfig):
    """
    Attach LoRA adapters to each model part (PP stage or whole model).
    Assumes weights are already initialized (NOT meta).
    """
    if not config.lora.enable_lora:
        return model_parts

    lora_cfg = LoraConfig(
        r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.lora_target_modules,
    )

    new_parts = []
    for stage_idx, m in zip(config.parallelism.stages_list, model_parts):
        m.requires_grad_(False) # 1) base 파라미터 전체 freeze
        _install_peft_hf_stubs(m)
        m = get_peft_model(m, lora_cfg) # 2) LoRA 주입

        s_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        s_all = sum(p.numel() for p in m.parameters())
        n_train = sum(1 for p in m.parameters() if p.requires_grad)
        n_all = sum(1 for p in m.parameters())
        logger.info(f"[LoRA] [Stage {stage_idx}] trainable num params: {n_train} / {n_all} = {100 * n_train / n_all:.2f}% (size: {s_train:,d} / {s_all:,d} = {100 * s_train / s_all:.2f}%)")

        new_parts.append(m)
    return new_parts


def _install_peft_hf_stubs(m: torch.nn.Module):
    # 1) prepare_inputs_for_generation
    if not hasattr(m, "prepare_inputs_for_generation"):
        def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
            if input_ids is None and "inputs" in kwargs:
                input_ids = kwargs["inputs"]
            return {"input_ids": input_ids, **kwargs}
        m.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, m)

    # 2) config: PEFT는 dict-like(.get) 기대 → dict로 주는 게 제일 간단
    if not hasattr(m, "config") or m.config is None:
        m.config = {}
    elif not hasattr(m.config, "get"):
        # 기존에 뭔가 있는데 .get이 없으면 dict로 감싸기
        m.config = {"_wrapped_config": m.config}

    # PEFT가 확인하는 키들 기본값 세팅
    m.config.setdefault("tie_word_embeddings", False)
    m.config.setdefault("use_cache", False)

    return m