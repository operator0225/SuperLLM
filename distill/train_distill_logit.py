"""2b단계: cross-tokenizer 로짓 증류 (Gemma 교사 → LFM2.5 학생).

Gemma(vocab 262,144)와 LFM2(vocab 65,536)는 토크나이저가 완전히 다르다. 따라서
같은 vocab을 전제로 하는 고전적 token-level KL(로짓 차원 1:1 매칭)은 불가능하다.
대신 **ULD**(Universal Logit Distillation, Boizard et al., NeurIPS 2024)를 쓴다:
각 위치의 확률분포를 내림차순 정렬해 L1 거리를 재는 vocabulary-agnostic 손실.

위치 정렬: 두 토크나이저의 offset mapping으로, "같은 문자 위치에서 시작하는 다음
토큰"을 예측하는 스텝끼리 매칭한다(근사). 정렬 안 되는 스텝은 건너뛴다.

총손실 = alpha * CE(hard, 학생 토큰 기준) + (1-alpha) * ULD(soft)
교사는 4-bit(NF4)로 로드 가능 — 질문의 "교사 4비트"가 여기다.

주의: GPU 검증이 필요한 실험성 코드다. 안전한 기본은 sequence-level 증류
(train_distill.py). micro-batch=1(정렬이 예제별) + grad accumulation 로 학습한다.

이어하기(resume): save_steps 마다 output_dir 에 어댑터+옵티마이저+진행상태를 저장한다.
output_dir 를 Google Drive 경로로 두면 Colab 세션을 여러 번 나눠 실행해도 이어진다.
재실행 시 trainer_state.json 이 있으면 자동으로 그 지점부터 재개한다.

사용:  python distill/train_distill_logit.py --config distill/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_step_alignment(teacher_offsets, student_offsets):
    """예측 스텝 정렬.

    스텝 i(위치 i에서 토큰 i+1 예측)의 타깃 토큰이 시작하는 문자 오프셋을 키로,
    (student_step, teacher_step) 쌍을 만든다. 같은 문자 오프셋에서 시작하는 타깃을
    예측하는 스텝끼리 매칭한다. offsets 는 (start, end) 리스트.
    """
    # 타깃 토큰 k(>=1)의 시작 문자 → 이를 예측하는 스텝 k-1
    t_start_to_step = {}
    for k in range(1, len(teacher_offsets)):
        start = teacher_offsets[k][0]
        t_start_to_step.setdefault(start, k - 1)

    pairs = []
    for i in range(1, len(student_offsets)):
        start = student_offsets[i][0]
        if start in t_start_to_step:
            pairs.append((i - 1, t_start_to_step[start]))
    return pairs


def uld_loss(student_logits, teacher_logits, temperature, top_k):
    """ULD: 정렬된 확률분포 간 L1 거리. vocab 크기 무관.

    student_logits: (P, V_s), teacher_logits: (P, V_t)  — P=정렬된 스텝 수.
    각 분포를 softmax(/T) 후 내림차순 정렬, top_k 로 truncate 하여 같은 길이로
    맞춘 뒤 L1. (top_k truncation 은 연산량을 위한 근사.)
    """
    q = F.softmax(student_logits.float() / temperature, dim=-1)
    p = F.softmax(teacher_logits.float() / temperature, dim=-1)
    k = min(top_k, q.size(-1), p.size(-1))
    q_top = torch.topk(q, k, dim=-1).values  # 이미 내림차순
    p_top = torch.topk(p, k, dim=-1).values
    return (q_top - p_top).abs().sum(dim=-1).mean()


def save_checkpoint(student, opt, tok, out_dir, epoch, index, step, lc):
    """어댑터(또는 전체 모델) + 옵티마이저 + 진행상태를 out_dir 에 저장.
    out_dir 가 Google Drive 경로면 Colab 세션이 끊겨도 다음 세션에서 이어진다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    torch.save(opt.state_dict(), out_dir / "optimizer.pt")
    (out_dir / "trainer_state.json").write_text(
        json.dumps({"epoch": epoch, "index": index, "global_step": step,
                    "student_load_in_4bit": bool(lc.get("student_load_in_4bit"))}),
        encoding="utf-8")
    print(f"[ckpt] epoch={epoch} index={index} step={step} → {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="distill/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scfg = cfg["student"]
    lc = cfg["distill_logit"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 체크포인트/이어하기 — Colab 분할 실행 지원. output_dir 를 Google Drive 경로로 두면
    # 세션이 끊겨도 다음 세션에서 이어서 학습한다.
    out_dir = Path(lc["output_dir"])
    state_path = out_dir / "trainer_state.json"
    resume_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    has_adapter = (out_dir / "adapter_config.json").exists()

    # --- 학생 ---
    # student_load_in_4bit=true 이면 QLoRA: 4-bit NF4 로 얼려 로드하고 LoRA 어댑터만 학습.
    # (4-bit 가중치는 직접 역전파 불가 → 어댑터로 학습. 교사·학생 모두 4-bit면 저VRAM.)
    s_tok = AutoTokenizer.from_pretrained(scfg["base_model"])
    if lc.get("student_load_in_4bit"):
        from transformers import BitsAndBytesConfig
        from peft import (LoraConfig, PeftModel, get_peft_model,
                          prepare_model_for_kbit_training)
        s_bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            scfg["base_model"], quantization_config=s_bnb,
            torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"": 0},
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        if resume_state and has_adapter:
            student = PeftModel.from_pretrained(base, str(out_dir), is_trainable=True)
            print(f"[resume] LoRA 어댑터 로드 ← {out_dir}")
        else:
            student = get_peft_model(base, LoraConfig(
                r=lc.get("lora_r", 16), lora_alpha=lc.get("lora_alpha", 32),
                lora_dropout=lc.get("lora_dropout", 0.05), bias="none",
                task_type="CAUSAL_LM", target_modules="all-linear",  # LFM2 모듈명 자동 탐지
            ))
        student.print_trainable_parameters()
    else:
        src = str(out_dir) if (resume_state and (out_dir / "config.json").exists()) else scfg["base_model"]
        student = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device)
    student.config.use_cache = False
    student.train()

    # --- 교사 (Gemma, freeze) ---
    # 교사는 forward만 하므로 양자화는 학습성이 아니라 '로짓 충실도'에만 영향.
    # teacher_precision: nf4(저VRAM) | int8(중간) | bf16(최상 충실도). E4B는 작아 여유 있으면 상향.
    t_tok = AutoTokenizer.from_pretrained(lc["teacher_model"])
    prec = lc.get("teacher_precision", "nf4" if lc.get("load_in_4bit") else "bf16")
    t_kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if prec == "nf4":
        from transformers import BitsAndBytesConfig
        t_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    elif prec == "int8":
        from transformers import BitsAndBytesConfig
        t_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif prec == "bf16":
        pass  # 전체 bf16 — 로짓 충실도 최상
    else:
        raise ValueError(f"unknown teacher_precision: {prec!r} (nf4|int8|bf16)")
    teacher = AutoModelForCausalLM.from_pretrained(lc["teacher_model"], **t_kwargs)
    print(f"[info] teacher precision = {prec}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 참고: Gemma4 등 멀티모달 config 는 최상위에 vocab_size 가 없다(text_config 중첩).
    # ULD 는 config 를 안 쓰므로 로그는 토크나이저 길이로 표기한다.
    if s_tok.get_vocab() == t_tok.get_vocab():
        print("[info] 토크나이저가 동일 — 정석 full-KL도 가능하지만 여기선 ULD로 진행.")
    else:
        print(f"[info] cross-tokenizer: student vocab≈{len(s_tok)}, "
              f"teacher vocab≈{len(t_tok)} → ULD 사용.")

    texts = [
        json.loads(l)["prompt"] + "\n" + json.loads(l)["response"]
        for l in Path(lc["target_data"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(lc["lr"]))
    T = lc["temperature"]
    alpha = lc["alpha_hard"]
    accum = lc["grad_accum"]
    max_len = lc["max_seq_len"]
    save_steps = lc.get("save_steps", 200)

    start_epoch, start_index, step = 0, 0, 0
    opt_path = out_dir / "optimizer.pt"
    if resume_state and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location=device))
        start_epoch = resume_state["epoch"]
        start_index = resume_state["index"]
        step = resume_state["global_step"]
        print(f"[resume] epoch={start_epoch} index={start_index} step={step} 부터 재개")

    for epoch in range(start_epoch, lc["epochs"]):
        first = start_index if epoch == start_epoch else 0
        for idx in range(first, len(texts)):
            text = texts[idx]
            s_enc = s_tok(text, return_offsets_mapping=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
            t_enc = t_tok(text, return_offsets_mapping=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
            pairs = build_step_alignment(
                t_enc["offset_mapping"][0].tolist(),
                s_enc["offset_mapping"][0].tolist(),
            )
            if not pairs:
                continue

            s_ids = s_enc["input_ids"].to(device)
            t_ids = t_enc["input_ids"].to(device)

            s_out = student(input_ids=s_ids).logits[0]              # (Ls, V_s)
            with torch.no_grad():
                t_out = teacher(input_ids=t_ids).logits[0].float()  # (Lt, V_t)

            # hard CE: 학생 자기 토크나이즈 기준 다음 토큰 예측 (= sequence-level KD)
            ce = F.cross_entropy(s_out[:-1], s_ids[0, 1:])

            s_steps = torch.tensor([i for i, _ in pairs], device=device)
            t_steps = torch.tensor([j for _, j in pairs], device=device)
            soft = uld_loss(s_out.index_select(0, s_steps),
                            t_out.index_select(0, t_steps).to(device),
                            T, lc["uld_top_k"])

            loss = (alpha * ce + (1 - alpha) * soft) / accum
            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % lc["logging_steps"] == 0:
                print(f"e{epoch} i{idx} step{step} loss={loss.item()*accum:.4f} "
                      f"ce={ce.item():.4f} uld={soft.item():.4f} aligned={len(pairs)}")
            step += 1
            if step % save_steps == 0:
                save_checkpoint(student, opt, s_tok, out_dir, epoch, idx + 1, step, lc)

    save_checkpoint(student, opt, s_tok, out_dir, lc["epochs"], 0, step, lc)
    print(f"done → {out_dir} "
          f"({'LoRA adapter' if lc.get('student_load_in_4bit') else 'full model'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
