import mlp_classifier as mlp
import lr_classifier as lrc
import argparse

def get_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_type','-c',
                        choices = ["mlp", "lrc"],
                        help = "Type of classifier to execute",
                        required = True
    )
    parser.add_argument('--train','-t',
                        help = "Specify whether or not to train model",
                        choices = ["y", "n"]
    )
    parser.add_argument('--input_sentence','-i',
                        help = "Input sentence to analyse",
                        type = str
    )
    return parser.parse_args()

def main():
    get_program_args()

if __name__== "__main__":
    args = get_program_args()
    if args.classification_type == "mlp" and args.train == "y":
        mlp.train_mlp()
    elif args.classification_type == "mlp" and args.train == "n":
        mlp.eval_sentence(args.input_sentence)
    elif args.classification_type == "lrc" and args.train == "y":
        lrc.train_lrc()
    elif args.classification_type == "lrc" and args.train == "n":
        lrc.eval_sentence(args.input_sentence)