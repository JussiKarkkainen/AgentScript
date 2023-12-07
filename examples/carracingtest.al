{
    "perception": {
        "vision_resolution": [
            1,
            3,
            64,
            64
        ],
        "hidden_dim": 32,
        "enc_kernel_size": 4,
        "dec_kernel_size": 5,
        "text": null,
        "audio": null
    },
    "world_model": {
        "input_size": 32,
        "hidden_size": 256,
        "num_cells": 1,
        "num_gaussians": 5,
        "mdn_output_size": 32
    },
    "actor": {
        "input_size": 64,
        "action_space": "discrete",
        "actions": 5,
        "num_mlp_layers": 1,
        "mlp_input_dim": 64,
        "mlp_hidden_dim": 40,
        "mlp_output_dim": 5
    },
    "data_config": {
        "num_episodes": 5,
        "max_frames": 10,
        "env": "CarRacing-v2",
        "policy": "random"
    },
    "meta": {
        "train": true,
        "make_dataset": true,
        "weights_path": "tests/weights/",
        "dataset_path": "tests/datasets/"
    }
}
