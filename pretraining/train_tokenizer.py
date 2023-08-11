import lzma
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import argparse

TOKENIZER_DIR = '../data/tokenizer-sp-hf'
DEFAULT_FILE_PATH = "../data/dataset_raw.txt.xz"
def main():
    """ set default hyperparams in default_hyperparams.py """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--vocab_size', default=32000)
    parser.add_argument('--file_path', default=DEFAULT_FILE_PATH)
    config = parser.parse_args()

    tokenizer = Tokenizer(models.Unigram())

    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.UnigramTrainer(
        vocab_size=config.vocab_size, special_tokens=special_tokens, unk_token="[UNK]"
    )

    print('Train tokenizer')
    file_extension = config.file_path.split(".")[-1]
    if file_extension == 'txt':
        tokenizer.train([], trainer=trainer)
    elif file_extension == 'xz':
        with lzma.open(config.file_path, mode='rt', encoding='utf-8') as f:
            tokenizer.train_from_iterator(f, trainer=trainer)

    print('Train finished')
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.decoder = decoders.Metaspace()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=512,
        padding_side="right",
        truncation_side="right",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    wrapped_tokenizer.save_pretrained(TOKENIZER_DIR)
    print('Tokenizer saved')

    # re-load tokenizer and test
    reloaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokens = reloaded_tokenizer.tokenize("Héllò hôw are ü?" * 5, add_special_tokens=False)
    print('Reloaded Tokenizer: ', tokens)


if __name__ == '__main__':
    main()
