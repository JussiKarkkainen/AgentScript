import wandb



class WandbLogger:
    def __init__(self, project_name: str, lr: float, arch: str, epochs: int):
        self.project_name = project_name
        self.lr = lr
        self.architecture = arch
        self.epochs = epochs

    def __enter__(self):
        wandb.init(
            project=project_name,

            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": self.architecture,
            "epochs": self.epochs,
            }
        )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

