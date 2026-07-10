"""2단계: sequence-level 지식 증류(= 교사 데이터로 학생 SFT).

teacher_data.jsonl 의 (prompt, response) 쌍을 chat 형식으로 만들어
LFM2.5-350M (bf16) 을 SFT 한다. 손실은 assistant 토큰에만 걸린다.

이것이 닫힌 API 교사에 대한 현실적 증류 방식이다(토크나이저 불일치 + top-20 제약으로
전체 logit KL 은 불가). 반드시 학습 가능한 bf16 체크포인트로 학습하고, q4_k_m 변환은
학습 이후 배포 단계(export_gguf.md)에서 수행한다.

사용:  python distill/train_distill.py --config distill/config.yaml   (CUDA GPU 권장)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def build_messages(example, system_prompt):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": example["prompt"]})
    msgs.append({"role": "assistant", "content": example["response"]})
    return {"messages": msgs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="distill/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scfg, dcfg, tcfg = cfg["student"], cfg["data"], cfg["train"]

    tok = AutoTokenizer.from_pretrained(scfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        scfg["base_model"],
        torch_dtype=torch.bfloat16 if tcfg.get("bf16", True) else torch.float32,
        attn_implementation="eager",  # LFM2 하이브리드; flash-attn 있으면 교체
    )

    ds = load_dataset("json", data_files=dcfg["teacher_out"], split="train")
    ds = ds.map(
        lambda ex: build_messages(ex, dcfg.get("system_prompt")),
        remove_columns=ds.column_names,
    )

    sft_config = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["epochs"],
        learning_rate=float(tcfg["lr"]),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg.get("weight_decay", 0.0),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        packing=tcfg.get("packing", True),
        assistant_only_loss=tcfg.get("assistant_only_loss", True),
        max_length=scfg["max_seq_len"],
        logging_steps=tcfg.get("logging_steps", 10),
        save_steps=tcfg.get("save_steps", 200),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    tok.save_pretrained(tcfg["output_dir"])
    print(f"saved → {tcfg['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
