# Distillation Pipeline — Gemini(교사) → LFM2.5-350M(학생)

닫힌 API 교사(Gemini)로 소형 학생 모델(LFM2.5-350M)을 증류하는 파이프라인.

## 왜 이 구조인가 (읽고 시작할 것)

### 1) q4_k_m은 학습 대상이 아니라 배포 결과물
`q4_k_m`은 llama.cpp의 **추론 전용** k-quant 포맷이라 역전파가 불가능하다.
학습은 반드시 **bf16 safetensors 체크포인트**(`LiquidAI/LFM2.5-350M`)로 하고,
학습이 끝난 뒤 GGUF로 변환 → `q4_k_m` 양자화한다.

```
bf16 base  ──(distill 학습)──▶  bf16 checkpoint  ──(convert_hf_to_gguf)──▶  f16 gguf  ──(llama-quantize)──▶  q4_k_m gguf
```

### 2) Gemini로는 고전적(logit) 증류가 불가능 → sequence-level KD 사용
- Gemini는 **top-20 logprobs만** 노출한다(`response_logprobs=true`, `logprobs≤20`).
  전체 vocab 분포를 주지 않으므로 전체 KL 증류가 불가능.
- Gemini와 LFM2의 **토크나이저가 다르다.** 토큰 단위 KL은 같은 vocab을 전제로 하므로
  성립하지 않는다(교차 토크나이저 증류 ULD/MinED는 복잡·손실적이라 후순위).
- 따라서 **black-box / sequence-level KD**: 교사가 만든 고품질 응답·CoT를 데이터로
  삼아 학생을 SFT한다. (선택) top-20 logprobs는 보조 신호로 저장만 해 둔다.

### 3) 교사를 오픈 웨이트로 바꿀 때: 토크나이저 불일치 → ULD
Gemini 대신 **Gemma 4 E2B/E4B**(오픈 웨이트)를 교사로 쓰면 전체 로짓에 접근할 수
있다. 하지만 vocab이 다르다:

| | vocab | 토크나이저 |
|---|---|---|
| Gemma 4 (E2B/E4B) | 262,144 | SentencePiece |
| LFM2.5 (230M/350M/1.2B) | 65,536 | LFM2 자체 |

고전적 full-logit KL은 **같은 vocab**을 전제로 하므로 성립하지 않는다. cross-tokenizer
증류 — **ULD(Universal Logit Distillation, Boizard et al. 2024)** 를 쓴다: 각 위치의
확률분포를 정렬해 L1 거리를 재는 vocab-무관 손실. 위치는 offset mapping으로 같은
문자 구간을 예측하는 토큰끼리 매칭한다. 구현: `train_distill_logit.py`.

**저VRAM 구성(권장): 교사·학생 둘 다 4-bit.**
- 교사(Gemma 4 E4B): NF4 로드(`load_in_4bit`). 로짓 신호로 충분.
- 학생(LFM2.5-350M): **QLoRA** — 4-bit NF4로 얼려 로드하고 LoRA 어댑터만 학습
  (`student_load_in_4bit`). 4-bit 가중치는 직접 역전파 불가하므로 어댑터로 학습한다.
- 둘 다 4-bit면 ~8GB급 GPU에서도 동작. 학습 후 어댑터를 병합→재양자화해 배포.

**LiteRT는 교사에 쓰지 않는다.** LiteRT/MediaPipe LLM API는 온디바이스 *생성*용이라
ULD에 필요한 위치별 262k full-logit 을 내주지 않고(기껏 top-k), GPU 학습 루프의
교사로도 부적합하다. LiteRT의 자리는 **최종 학생 배포**(DESIGN §8 NPU 경로)다.

참고: LFM2.5 최소 크기는 **230M**(`LiquidAI/LFM2.5-230M`, 2026-06). 350M/1.2B도 있음.

### 4) 라이선스
Google Gemini API 약관은 출력물로 경쟁 모델을 학습하는 것을 제한하는 조항이 있다.
배포 전 최신 약관을 확인할 것. 연구/개인 실험과 상용 배포는 취급이 다르다.

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 1. 교사 데이터 생성 | `gen_teacher_data.py` | `data/seed_prompts.jsonl` | `data/teacher_data.jsonl` |
| 2a. sequence-level 증류 | `train_distill.py` | teacher_data + bf16 base | `out/…-distill/` |
| 2b. cross-tokenizer 로짓 증류(ULD) | `train_distill_logit.py` | 대상 텍스트 + Gemma 교사(4bit) + bf16 base | `out/…-uld/` |
| 3. GGUF 배포 변환 | `export_gguf.md` | bf16 checkpoint | `…-q4_k_m.gguf` |

- **2a (기본, 안전)**: 교사 응답으로 학생 SFT. 교사가 닫힌 API(Gemini)여도 됨. 견고함.
- **2b (강한 신호, 실험)**: Gemma 교사의 출력 분포를 ULD로 증류. 오픈 웨이트 교사 필요,
  GPU 필요. 2a보다 강하지만 정렬 근사가 들어감 — GPU 검증 후 사용.

## 실행

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...          # aistudio.google.com/apikey

# 1) 교사 데이터 생성 (seed_prompts.jsonl 에 원하는 프롬프트를 채운 뒤)
python distill/gen_teacher_data.py --config distill/config.yaml

# 2) 증류 학습 (GPU 필요)
python distill/train_distill.py --config distill/config.yaml

# 3) 배포용 GGUF q4_k_m 변환 → distill/export_gguf.md 참고
```

## 데이터 스케일 감각
- 스모크 테스트: seed 수십 개로 파이프라인 동작 확인.
- 의미 있는 능력 이전: 도메인당 수만~수십만 개의 교사 응답. 다양성과 난이도 분포가
  최종 품질을 좌우한다. 추론 능력 이전은 **CoT(사고 과정) 포함 응답**이 핵심.
