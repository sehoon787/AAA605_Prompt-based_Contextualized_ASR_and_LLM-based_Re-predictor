config = {
    "speech_input_dim": 80,
    "speech_hidden_dim": 256,
    "num_stages": 4,
    "blocks_per_stage": 2,
    "reduction_factor": 2,

    "pretrained_model_name": "bert-base-uncased",
    "bert_hidden_dim": 768,
    "adapter_dim": 256,
    "fusion_dim": 256,

    "vocab_size": 30522,  # BERT tokenizer vocab 그대로 사용
    "embed_dim": 256,
    "predictor_hidden_dim": 512,
    "encoder_output_dim": 256,
    "joint_dim": 512
}
