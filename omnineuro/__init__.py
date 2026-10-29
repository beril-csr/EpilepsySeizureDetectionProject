"""OmniNeuro -> memristive Cellular Automaton: explainable EEG seizure-likeness feeding a brain-tissue model."""
from .config import Config
from .pipeline import analyse, evaluate

__all__ = ["Config", "analyse", "evaluate"]
