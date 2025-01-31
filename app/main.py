import os
import numpy as np
import torch
from transformers import T5Tokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


from configuration.api_config import BaseConfig
from indexer import TokenIndexer
from data_preparation import AVFeatures
from dataset import ConcatDataset
from models.t5_model import Classifier

from pydantic import BaseModel, validator, ValidationError
from fastapi import FastAPI, BackgroundTasks, Response, Depends
from logger import get_logger

logger = get_logger()
BASE_STORAGE_PATH = os.environ.get("STORAGE_PATH", "users")

tags_metadata = [
    {
        "name": "train",
        "description": "Trigger training manually",
    },
    {
        "name": "infer",
        "description": "Infer with a list of messages",
    },
]

app = FastAPI(title="T5 Authorship Verification", openapi_tags=tags_metadata)

ARGS = BaseConfig()
test_samples = 100

logger.debug(ARGS.best_model_path)

# ------------------------------ Load T5 Tokenizer ---------------------------
T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_path)

# ----------------------------- Load Model -----------------------------------
MODEL = Classifier.load_from_checkpoint(ARGS.best_model_path)


# live api check if server is alive
@app.get("/live", tags=["live"])
async def aa_live():
    return {"message": "ok"}, 200


class InferRequest(BaseModel):
    text1: str
    text2: str
    negative_threshold: float = None

    @validator('negative_threshold', pre=True, always=True)
    def check_threshold(cls, value):
        if value is not None and (value <= 0 or value >= 1):
            raise ValidationError('negative_threshold should be between 0 and 1 (exclusive)')
        return value


@app.post("/inference", tags=["Inference"])
async def infer(request: InferRequest):
    try:
        text1 = request.text1
        text2 = request.text2
        threshold = 1 - request.negative_threshold if request.negative_threshold else 0.1
        FIRST_TEXT = [text1]
        SECOND_TEXT = [text2]

        av_features_obj = AVFeatures(
            datasets=[FIRST_TEXT, SECOND_TEXT],
            tokenizer=word_tokenize,
            pos_tagger=pos_tag)

        FIRST_TEXT_FEATURES, SECOND_TEXT_FEATURES = av_features_obj()
        POS_INDEXER = TokenIndexer()
        POS_INDEXER.load(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.pos2index_file),
                         idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2pos_file))

        FIRST_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(FIRST_TEXT_FEATURES[0])
        SECOND_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(SECOND_TEXT_FEATURES[0])

        # ---------------------------- Prepare of aggregated data -------------------------------
        COLUMNS2DATA = {"first_text": FIRST_TEXT,
                        "second_text": SECOND_TEXT,
                        "first_punctuations": FIRST_TEXT_FEATURES[1],
                        "second_punctuations": SECOND_TEXT_FEATURES[1],
                        "first_information": FIRST_TEXT_FEATURES[2],
                        "second_information": SECOND_TEXT_FEATURES[2],
                        "first_pos": FIRST_TEXT_FEATURES_POS,
                        "second_pos": SECOND_TEXT_FEATURES_POS}

        # ------------------------- Create dataloader -----------------------------------
        DATASET = ConcatDataset(data=COLUMNS2DATA,
                                tokenizer=T5_TOKENIZER,
                                max_len=ARGS.max_len)

        DATALOADER = torch.utils.data.DataLoader(DATASET, batch_size=1, shuffle=False, num_workers=1)
        PREDICTIONS = []
        PROBABILITIES = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i_batch, sample_batched in enumerate(DATALOADER):
            print("Batch: {}".format(i_batch))

            sample_batched = {k: v.to(device) for k, v in sample_batched.items()}

            OUTPUT = MODEL(sample_batched)
            OUTPUT = torch.softmax(OUTPUT, dim=1)
            OUTPUT_probs = OUTPUT[:, 1].detach().cpu().numpy()  # Getting the probability of the positive class
            PREDICTIONS.append(OUTPUT_probs)
            predictions = np.concatenate(PREDICTIONS)

            binary_predictions = (predictions >= threshold).astype(int).tolist()
            predictions = predictions.tolist()

            return {"label": binary_predictions[0], "probability": 1 - predictions[0]}

    except RuntimeError as error:
        # Check for CUDA out of memory error
        torch.cuda.empty_cache()
        logger.error(error)
        return {"error": str(error)}


