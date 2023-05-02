import pickle
import typing as typ

from convokit import Corpus, download
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
from tqdm import tqdm
import tiktoken
from pathlib import Path
from seed import seed_everything

seed_everything(181993)

input_length: int = 32

# OSX needs hardcode full path ~/ is not work.
corpus_dir = f"{Path.home()}/.convokit/downloads/movie-corpus"
if os.path.exists(corpus_dir):
    print("Cornell Corpus exists. Reading from disk.")
    corpus = Corpus(filename=corpus_dir)
else:
    print("Cornel Corpus not exists. Downloading and writing to disk.")
    corpus = Corpus(filename=download("movie-corpus"))

conversations = list(corpus.iter_conversations())
enc = tiktoken.get_encoding("gpt2")

def clean_encoded_input(_input: typ.List[int]) -> typ.List[int]:
    """Padding or truncating the _input."""
    if len(_input) > input_length:
        ans = _input[:input_length]
    else:
        need_padding = input_length - len(_input)
        ans = _input + [0] * need_padding
    return ans

input_text: typ.List[typ.List[int]] = []
target_text: typ.List[typ.List[int]] = []
offset: int = 100
for conversation in tqdm(conversations[:offset], total=len(conversations[:offset])):
    utterances = list(conversation.iter_utterances())
    for i in range(len(utterances) - 1):
        input_enc_string: typ.List[int] = enc.encode_ordinary(utterances[i].text)
        input_text.append(clean_encoded_input(input_enc_string))
        target_enc_string: typ.List[int] = enc.encode_ordinary(utterances[i+1].text)
        target_text.append(clean_encoded_input(target_enc_string))
data_dict = {
    "enc_prompt": input_text,
    "enc_response": target_text
}
df = pd.DataFrame.from_dict(data_dict)
# df["prompt_length"] = df.apply(lambda row: len(row.enc_prompt), axis=1)
# df["response_length"] = df.apply(lambda row: len(row.enc_response), axis=1)
# df.to_csv("./data/cornel-movie/encoded.csv", index=False)
table = pa.Table.from_pandas(df)
pq.write_table(table, "./data/cornel-movie/data.arrow")
