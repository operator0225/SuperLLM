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

### 3) LFM2.5에 230M은 없다
최소 크기는 **350M**. (`LiquidAI/LFM2.5-350M`)

### 4) 라이선스
Google Gemini API 약관은 출력물로 경쟁 모델을 학습하는 것을 제한하는 조항이 있다.
배포 전 최신 약관을 확인할 것. 연구/개인 실험과 상용 배포는 취급이 다르다.

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 1. 교사 데이터 생성 | `gen_teacher_data.py` | `data/seed_prompts.jsonl` | `data/teacher_data.jsonl` |
| 2. 증류 학습(SFT) | `train_distill.py` | teacher_data + bf16 base | `out/…-distill/` |
| 3. GGUF 배포 변환 | `export_gguf.md` | bf16 checkpoint | `…-q4_k_m.gguf` |

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
