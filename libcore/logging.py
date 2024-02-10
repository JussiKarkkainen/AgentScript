import wandb
from typing import Any, Dict

class Logger:
    def __init__(self, project_name: str, config: Dict[str, Any], auto_log: bool = True):
        self.project_name = project_name
        self.config = config
        self.auto_log = auto_log
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            wandb.init(project=self.project_name, config=self.config)
            self.initialized = True

    def log(self, metrics: Dict[str, Any], step: int = None):
        if self.auto_log and self.initialized:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.initialized:
            wandb.finish()

