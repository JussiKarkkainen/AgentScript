import wandb

class WandbLogger:
    def __init__(self, config):
        self.user_config = config
        self.wandb_config = self.create_wandb_config()
    
    def create_wandb_config(self):
        def extract_numerics(source_dict, target_dict):
            for key, value in source_dict.items():
                if isinstance(value, (int, float)):
                    target_dict[key] = value
                elif isinstance(value, dict):
                    extract_numerics(value, target_dict)

        config = {}
        for conf in self.user_config["logs"]:
            if conf in self.user_config:
                if isinstance(self.user_config[conf], (int, float)):
                    config[conf] = self.user_config[conf]
                elif isinstance(self.user_config[conf], dict):
                    extract_numerics(self.user_config[conf], config)

        return config
    
    def log(self, logs):
        wandb.log(logs)

    def __enter__(self):
        wandb.init(
            project=self.user_config["type"],

            config=self.wandb_config
        )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

