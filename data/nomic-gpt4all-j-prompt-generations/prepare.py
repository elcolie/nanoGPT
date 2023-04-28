import typing as typ
import pandas as pd
import pyarrow as pa
import torch
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
from transformers import AutoModelForCausalLM

dataset = load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision="v1.2-jazzy")
train_dataset = dataset["train"]
n = len(train_dataset)  # 711126
# data_size = int(0.00005 * n)
data_size = 100
print(f"data_size: {data_size}")
desired_length = 64  # Maximum length

enc = tiktoken.get_encoding("gpt2")
prompt_list: typ.List[typ.List[int]] = []
response_list: typ.List[typ.List[int]] = []
try:
    for index, val in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        if index < data_size:
            encoded_prompt = enc.encode_ordinary(val["prompt"])
            padding_length = desired_length - len(encoded_prompt)
            padded_tensor = encoded_prompt + [0] * padding_length
            prompt_list.append(padded_tensor[:desired_length])

            encoded_response = enc.encode_ordinary(val["response"])
            res_padding_length = desired_length - len(encoded_response)
            res_padded_tensor = encoded_response + [0] * res_padding_length
            response_list.append(res_padded_tensor[:desired_length])
        else:
            break
    data_dict = {
        "enc_prompt": prompt_list,
        "enc_response": response_list,
    }
    df = pd.DataFrame.from_dict(data_dict)
    table = pa.Table.from_pandas(df)

    # Write the Arrow format to a file
    with pa.output_stream('data.arrow') as f:
        writer = pa.RecordBatchFileWriter(f, table.schema)
        writer.write_table(table)
        writer.close()

except Exception:
    import ipdb; ipdb.set_trace()
