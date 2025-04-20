# On-Device MNIST 추론 최적화 (TensorFlow Lite for Microcontrollers)

이 프로젝트는 Arduino 기반 임베디드 환경에서 MNIST 숫자 분류 모델의 추론 속도를 개선하기 위해  
Google의 [TensorFlow Lite for Microcontrollers (TFLM)](https://github.com/tensorflow/tflite-micro) 기반 연산 코드를 최적화한 작업입니다.

---

## 📌 프로젝트 목적

- ARM Cortex-M 계열 칩에서 MNIST 추론을 빠르게 수행하기 위한 연산 최적화
- TFLM 내 병목 연산자 분석 및 최적화 적용
- 루프 언롤링, SIMD 명령어, 컴파일러 플래그 등을 통한 성능 개선

---

## ⚙️ 주요 최적화 내용

- `arm_convolve_s8` → **3x3 전용 커스텀 함수**로 변경
- **SIMD 명령어 (`__SMLAD`, `__UQADD16` 등)** 적용
- **루프 언롤링 (loop unrolling)** 으로 반복 연산 최소화
- `arm_nn_vec_mat_mult_s8` 등 FC 레이어 연산 최적화
- 컴파일러 플래그 (`-O2`, `-fomit-frame-pointer`) 적용

---

## 📈 성능 개선 결과

| 단계 | 추론 시간 (μs) |
|------|-----------------|
| 초기 Float32 모델 | 89,293 |
| Int8 양자화 적용 | 29,690 |
| 루프 언롤링 & SIMD 적용 | 14,594 |
| 컴파일러 최적화 (O2) | **13,294** |

→ **최종 약 85% 성능 향상** 달성

---


