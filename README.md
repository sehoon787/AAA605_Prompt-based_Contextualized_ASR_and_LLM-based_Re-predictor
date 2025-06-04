# Full ASR + Re-Predictor 논문 재현 패키지

## ✅ 프로젝트 구조

- **ASRModel**: Zipformer 기반 Prompt-aware Encoder + RNNT Decoder
- **Pruned RNNT Loss**: 학습 속도 및 효율 최적화
- **Beam Search**: Inference에서 n-best 후보 추출
- **Re-Predictor**: LLM 기반 Re-Ranking (LoRA fine-tuning 포함)

## ✅ 설치

```bash

pip install -r requirements.txt
```

## ✅ Data Load
``` bash

python dataset/prepare_librispeech.py
``` 

1️⃣ ASR 모델 학습
``` bash

python trainer/train_asr.py
``` 

2️⃣ Fine-tuning 데이터 생성 (n-best 생성)
``` bash

python trainer/generate_re_predictor_data.py
``` 

3️⃣ Re-Predictor Fine-tuning (LoRA 적용)
``` bash

python trainer/fine_tune_re_predictor.py
``` 

4️⃣ 통합 Inference 파이프라인
``` bash

python inference_pipeline.py
``` 