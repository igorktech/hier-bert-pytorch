from hierbert_model.modelling_hierbert import HierBertModel, HierBertForMaskedLM
from hierbert_model.configuration_hierbert import HierBertConfig
from transformers import AutoTokenizer
import argparse


def main():
    parser = argparse.ArgumentParser(description="Base model generation")

    parser.add_argument("--output_dir", type=str, default="../data/model", help="Default model output directory")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="../data/tokenizer-sp",
                        help="Tokenizer name or directory")
    # TODO: add configuration options
    # parser.add_argument("--encoder_config", type=str, choices=["i", "ah", "ec","lc"],
    #                     help="Select encoder layers configuration:
    #                     i - interleaved,
    #                     ah - ad-hoc,
    #                     ec - early-contextualization
    #                     lc - late-contextualization")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Size of the vocabulary.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Dimensionality of the hidden layers.")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads in each attention layer.")
    parser.add_argument("--intermediate_size", type=int, default=2048,
                        help="Dimensionality of the intermediate (feedforward) layers.")
    parser.add_argument("--hidden_act", type=str, default="gelu", help="Activation function for the hidden layers.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
                        help="Dropout probability for the hidden layers.")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1,
                        help="Dropout probability for the attention layers.")
    parser.add_argument("--max_position_embeddings", type=int, default=512,
                        help="Maximum number of positional embeddings.")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="Size of the token type vocabulary.")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Initializer range for model weights.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6, help="Epsilon for layer normalization.")
    parser.add_argument("--norm_first", type=bool, default=True, help="PreLayer normalization.")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Token ID for padding.")
    parser.add_argument("--sep_token_id", type=int, default=3, help="Token ID for separating segments.")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Setup model config
    config = HierBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_act=args.hidden_act,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=args.initializer_range,
        layer_norm_eps=args.layer_norm_eps,
        norm_first=args.norm_first,
        pad_token_id=args.pad_token_id,
        sep_token_id=args.sep_token_id)

    # Setup blank model
    HierBertModel.register_for_auto_class("AutoModel")
    model = HierBertModel(config)

    # Save tokenizer and model
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    print(
        f'Hierarchical BERT model with {model.num_parameters() / 1e6:.2f}M parameters and tokenizer were generated and saved to {args.output_dir}')


if __name__ == '__main__':
    main()
