from configs.carracingtest import TestCarRacingConfig
import json


out_file = open("examples/carracing.al", "w")
json.dump(TestCarRacingConfig, out_file, indent=4)



