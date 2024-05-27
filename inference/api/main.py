import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
import tokenizers
from torch import nn

app = FastAPI()


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)  # For start and end logits
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state

        out_cat = torch.cat((sequence_output[:, -1, :], sequence_output[:, -2, :]), dim=-1)
        logits = self.l0(self.drop_out(out_cat))

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    BERT_PATH = "/app/bert-base-uncased"
    MODEL_PATH = "/app/api/model_0.bin"
    TRAINING_FILE = "/app/input/train_folds.csv"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )


model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
model_config.output_hidden_states = True
model = TweetModel(conf=model_config)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)

class InputData(BaseModel):
    text: str

def get_prediction_text(text, start_logits, end_logits):
    start_idx = torch.argmax(start_logits, dim=0).item()
    end_idx = torch.argmax(end_logits, dim=0).item()
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))
    return ' '.join(tokens[start_idx:end_idx + 1])

# Define a prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    try:
        inputs = tokenizer(
            data.text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=config.MAX_LEN
        )
        start_logits, end_logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids', None)
        )
        prediction_text = get_prediction_text(data.text, start_logits[0], end_logits[0])
        return {"prediction": prediction_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))