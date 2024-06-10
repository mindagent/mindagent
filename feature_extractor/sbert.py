import torch 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sentence_transformers import SentenceTransformer

class SBert(BaseFeaturesExtractor):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.eval()


    def forward(self, observations) -> torch.Tensor:
        with torch.no_grad():
            features = self.model.encode(observations)
        return features