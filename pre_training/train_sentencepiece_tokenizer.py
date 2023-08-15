import sentencepiece as spm
import lzma

from transformers import DebertaV2Tokenizer, AutoTokenizer
import argparse
import os

TOKENIZER_DIR = '../data/tokenizer-sp'
DEFAULT_FILE_PATH = "../data/datasets/raw/dataset_raw.txt"


def main():
    """ set default hyperparams in default_hyperparams.py """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--vocab_size', default=32000)
    parser.add_argument('--file_path', default=DEFAULT_FILE_PATH)
    config = parser.parse_args()

    file_extension = config.file_path.split(".")[-1]
    if file_extension == 'xz':
        with lzma.open(config.file_path, 'rb') as compressed_file, open(os.path.splitext(config.file_path)[0],
                                                                        'wb') as output:
            decompressed_data = compressed_file.read()
            output.write(decompressed_data)
            # Update path
            config.file_path = os.path.splitext(config.file_path)[0]

    spm.SentencePieceTrainer.Train(
        input=config.file_path,
        model_prefix='sp',
        vocab_size=config.vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        user_defined_symbols='[MASK]',
        model_type='unigram'
    )

    tokenizer_deberta = DebertaV2Tokenizer(
        vocab_file="../data/sentencepiece/sp.model",
        max_len=512,
    )

    tokenizer_deberta.save_pretrained(TOKENIZER_DIR)
    print('Tokenizer saved')

    # re-load tokenizer and test
    reloaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokens = reloaded_tokenizer.tokenize("Héllò hôw are ü?" * 5, add_special_tokens=False)
    print('Reloaded Tokenizer: ', tokens)


if __name__ == '__main__':
    main()
