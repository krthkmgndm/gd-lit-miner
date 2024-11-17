from pathlib import Path

# System constants
DEFAULT_BATCH_SIZE = 32
MPS_BATCH_SIZE = 32
CUDA_BATCH_SIZE = 64
CPU_BATCH_SIZE = 16

# Model constants
MODEL_NAME = "dmis-lab/biobert-v1.1"
MAX_SEQ_LENGTH = 512  # Standard BERT sequence length
MAX_BATCH_TOKENS = 32768  # Maximum tokens per batch
PAD_TOKEN_ID = 0
EMBEDDING_SIZE = 768  # BERT base hidden size
ATTENTION_HEAD_SIZE = 64
NUM_ATTENTION_HEADS = 12

# Processing constants
MIN_SENTENCE_LENGTH = 10
MAX_SENTENCE_LENGTH = 128
MIN_CONFIDENCE_SCORE = 0.5

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GENERATED_DATA_DIR = DATA_DIR / "generated"
REPORTS_DIR = PROJECT_ROOT / "reports"
