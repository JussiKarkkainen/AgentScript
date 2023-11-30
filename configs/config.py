CarRacingConfig = {
            "perception": {
                "vision_resolution": (1, 3, 64, 64),
                "hidden_dim": 32,
                "enc_kernel_size": 4,
                "dec_kernel_size": 5,
                "text": None,
                "audio": None
            },
            "world_model": {
                "input_shape": 32,
                "hidden_size": 256
            },
            "actor": {
                "input_size": 64,
                "action_space": "discrete",
                "actions": 5
            }
            "data_config": {
                "num_episodes": 10000,
                "env": "CarRacing-V2",
                "policy": "random"
            }
}
