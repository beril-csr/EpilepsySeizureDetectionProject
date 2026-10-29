"""STEP 6 - The whole pipeline in 10 lines.  This is all `python -m omnineuro demo` does."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omnineuro import Config, analyse, evaluate, outputs
from omnineuro.data import make_synthetic

data, fs = make_synthetic()
cfg = Config()                                  # try Config(memory=0.0) or Config(bias=2.0)
result = analyse(data, fs, cfg)
print(evaluate(result, [(500, 545)]))
outputs.dashboard(result, "outputs/steps/step6_dashboard.png", [(500, 545)])
print(outputs.make_report(result))
