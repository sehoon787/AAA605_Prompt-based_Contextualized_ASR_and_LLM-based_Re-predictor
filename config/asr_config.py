config = {
    "speech_input_dim": 80,
    "speech_hidden_dim": 256,
    "zipformer_blocks": [2, 2, 3, 4, 3, 2],
    "reduction_factors": [2, 3, 4, 3, 2],

    "pretrained_model_name": "bert-base-uncased",
    "bert_hidden_dim": 768,
    "adapter_dim": 256,
    "fusion_dim": 256,

    "vocab_size": 1024,  # 예: a-z + blank + space + apostrophe 등
    "embed_dim": 256,
    "encoder_output_dim": 256,
    "joint_dim": 512
}
