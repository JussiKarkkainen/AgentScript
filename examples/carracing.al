# Json config for World Models

{
    "perception": {
        "vision_resolution": [
            1,
            3,
            64,
            64
        ],
        "model": "CNN"
        "model_params": {
            "hidden_dim": 32,
            "enc_kernel_size": 4,
            "dec_kernel_size": 5,
            "text": null,
            "audio": null
        }
    },
    "world_model": {
        "model": "MDNLSTM",
        "model_params": {
            "input_shape": 32,
            "hidden_size": 256
        }
    },
    "actor": {
        "model": "MLP",
        "model_params": {
            "input_size": 64,
            "action_space": "discrete",
            "actions": 5
        }
    },
    "algorithm": "WorldModel",
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
