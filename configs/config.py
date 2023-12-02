CarRacingConfig = {
            "perception": {
                "vision_resolution": (1, 3, 64, 64),
                "hidden_dim": 32,
                "enc_kernel_size": 4,
                "dec_kernel_size": 5,
                "text": None,
                "audio": None,
            },
            "world_model": {
                "input_shape": 32,
                "hidden_size": 256,
            },
            "actor": {
                "input_size": 64,
                "action_space": "discrete",
                "actions": 5,
            },
            "data_config": {
                "num_episodes": 10000,
                "max_frames": 1000,          # Maximum frames/episode
                "env": "CarRacing-v2",
                "policy": "random"
            },
            "meta": {
                "train": True,
                "make_dataset": True,
                "weights_path": "weights/",
                "dataset_path": "datasets/"
            },
            "train": {
                "num_epochs": 10
            }
}

TestCarRacingConfig = {
            "perception": {
                "vision_resolution": (1, 3, 64, 64),
                "hidden_dim": 32,
                "enc_kernel_size": 4,
                "dec_kernel_size": 5,
                "text": None,
                "audio": None,
            },
            "world_model": {
                "input_shape": 32,
                "hidden_size": 256,
            },
            "actor": {
                "input_size": 64,
                "action_space": "discrete",
                "actions": 5,
            },
            "data_config": {
                "num_episodes": 5,
                "max_frames": 10,          # Maximum frames/episode
                "env": "CarRacing-v2",
                "policy": "random"
            },
            "meta": {
                "train": True,
                "make_dataset": True,
                "weights_path": "tests/weights/",
                "dataset_path": "tests/datasets/"
            }
}
