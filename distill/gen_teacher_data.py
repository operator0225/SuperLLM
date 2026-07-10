"""1단계: 교사(Gemini)로 증류 데이터 생성.

seed_prompts.jsonl 의 각 프롬프트에 대해 교사 응답을 받아 teacher_data.jsonl 로 저장한다.
(선택) top-k logprobs 도 함께 저장한다 — sequence-level SFT 에는 응답 텍스트만 쓰지만,
나중에 실험할 보조 신호로 남겨 둔다.

토크나이저 불일치 + top-20 제약 때문에 전체 logit KL 증류는 하지 않는다. 여기서
만든 (prompt, response) 쌍으로 학생을 SFT 하는 것이 이 파이프라인의 증류 방식이다.

사용:  python distill/gen_teacher_data.py --config distill/config.yaml
환경:  GEMINI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm


def load_done_prompts(out_path: Path) -> set[str]:
    """이미 처리한 프롬프트를 모아 재실행 시 이어서 진행(resume)."""
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def extract_logprobs(candidate) -> list | None:
    """candidate.logprobs_result 를 JSON 직렬화 가능한 형태로 변환. 없으면 None."""
    lpr = getattr(candidate, "logprobs_result", None)
    if lpr is None:
        return None
    steps = []
    for top in getattr(lpr, "top_candidates", []) or []:
        cands = getattr(top, "candidates", []) or []
        steps.append([
            {"token": getattr(c, "token", None), "logprob": getattr(c, "log_probability", None)}
            for c in cands
        ])
    return steps or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="distill/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg, dcfg = cfg["teacher"], cfg["data"]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        return 2

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: pip install google-genai", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)

    seed_path = Path(dcfg["seed_prompts"])
    out_path = Path(dcfg["teacher_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = [
        json.loads(l)["prompt"]
        for l in seed_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    done = load_done_prompts(out_path)
    todo = [p for p in prompts if p not in done]
    print(f"seed={len(prompts)} done={len(done)} todo={len(todo)}")

    gen_config = types.GenerateContentConfig(
        system_instruction=dcfg.get("system_prompt"),
        temperature=tcfg["temperature"],
        max_output_tokens=tcfg["max_output_tokens"],
        response_logprobs=tcfg.get("request_logprobs", False),
        logprobs=tcfg.get("top_logprobs") if tcfg.get("request_logprobs") else None,
    )

    written = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for prompt in tqdm(todo, desc="teacher"):
            resp = call_with_retry(
                client, tcfg["model"], prompt, gen_config,
                max_retries=tcfg.get("max_retries", 5),
            )
            if resp is None:
                continue
            text = (resp.text or "").strip()
            if not text:
                continue
            record = {"prompt": prompt, "response": text}
            if resp.candidates:
                lp = extract_logprobs(resp.candidates[0])
                if lp is not None:
                    record["teacher_logprobs"] = lp
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

    print(f"wrote {written} records → {out_path}")
    return 0


def call_with_retry(client, model, prompt, gen_config, max_retries=5):
    """지수 백오프 재시도. 실패 시 None."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=prompt, config=gen_config
            )
        except Exception as e:  # noqa: BLE001 - API 예외 종류가 다양
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{max_retries} in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
