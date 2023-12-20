# Json config for a Deep Q network

{
    "perception": {
        "vision_resolution": [
            4
        ],
        "model": "MLP",
        "model_params": {
            "hidden_dim": 32,
            "enc_kernel_size": 4,
            "dec_kernel_size": 5,
            "text": null,
            "audio": null
        }
    },
    "world_model": {
        "model": null
    },
    "actor": {
        "model": null
    },
    "algorithm": "DQN",
    "data_config": {
        "num_episodes": 5,
        "max_frames": 10,
        "env": "CartPole-v1",
        "policy": "random"
    },
    "meta": {
        "train": true,
        "make_dataset": true,
        "weights_path": "tests/weights/",
        "dataset_path": "tests/datasets/"
    }
}

