import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .utils.constants import (
    CUDA_BATCH_SIZE,
    EMBEDDING_SIZE,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    MPS_BATCH_SIZE,
)
from .utils.silicon_optimizer import M2Optimizer


class BiomedicalNLP:
    def __init__(self):
        """Initialize BioBERT model and tokenizer with M2 optimizations."""
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, max_length=MAX_SEQ_LENGTH
        )
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = M2Optimizer.get_device()
        self.batch_size = (
            MPS_BATCH_SIZE if self.device == torch.device("mps") else CUDA_BATCH_SIZE
        )
        self.model = self.model.to(self.device)
        M2Optimizer.optimize_memory_usage()

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, trunction=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)
            if self.device == torch.device("mps"):
                embeddings = embeddings.to("cpu")

        return embeddings
