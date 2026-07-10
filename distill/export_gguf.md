# 3단계: 학습된 체크포인트 → GGUF q4_k_m (배포용)

증류 학습이 끝난 bf16 체크포인트를 llama.cpp 로 변환·양자화한다.
**이 단계는 학습이 아니라 배포 패키징이다.** q4_k_m 은 여기서 처음 만들어진다.

## 준비
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j     # llama-quantize 빌드
pip install -r requirements.txt                              # convert 스크립트 의존성
```

## 변환
```bash
# (1) HF 체크포인트 → f16 GGUF
python convert_hf_to_gguf.py \
    ../distill/out/lfm2.5-350m-distill \
    --outfile ../distill/out/lfm2.5-350m-distill-f16.gguf \
    --outtype f16

# (2) f16 → q4_k_m 양자화
./build/bin/llama-quantize \
    ../distill/out/lfm2.5-350m-distill-f16.gguf \
    ../distill/out/lfm2.5-350m-distill-q4_k_m.gguf \
    Q4_K_M
```

## 확인
```bash
./build/bin/llama-cli -m ../distill/out/lfm2.5-350m-distill-q4_k_m.gguf \
    -p "테스트 프롬프트" -n 128
```

## 참고
- `convert_hf_to_gguf.py` 가 LFM2 아키텍처를 인식하려면 최신 llama.cpp 가 필요하다.
  변환 에러가 나면 llama.cpp 를 업데이트할 것.
- 350M 을 q4_k_m 로 양자화하면 대략 ~200MB 안팎 — 원래 §DESIGN 의 500MB 예산에 들어온다.
- 온디바이스(S25+ NPU) 배포는 별개 경로다. GGUF 는 llama.cpp(CPU/GPU)용이고,
  Hexagon NPU 로 올리려면 LiteRT + QNN delegate 파이프라인이 따로 필요하다(DESIGN §8).
