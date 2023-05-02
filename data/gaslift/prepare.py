"""
Prepare the dataset by download chp1 Burney book.
Burney's new book, Practical Optimization of Petroleum Production Systems
Convert to text by www.pdf2go.com
"""
import os

import numpy as np
import tiktoken
from pdfminer.high_level import extract_text
import pickle


def main() -> None:
    """Run main function."""
    # Read txt file from disk.
    data = extract_text(
        "data/gaslift/Fundamentals of gas lift engineering  well design and troubleshooting by Hernandez, Ali (z-lib.org).pdf")
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # train has 581,225 tokens
    # val has 64,775 tokens


if __name__ == "__main__":
    main()
