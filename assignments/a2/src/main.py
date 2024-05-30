import mlp_classifier as mlp
import logistic_regression as lrc
import argparse

def get_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_type','-c',
                        choices = ["mlp", "lr"],
                        help = "Type of classifier to execute",
                        required = True
    ),
    parser.add_argument('--input_sentence','-i',
                        help = "Input sentence to analyse",
                        type = str,
                        required = True
    )
    return parser.parse_args()

def main():
    get_program_args()

if __name__== "__main__":
    args = get_program_args()
    if args.classification_type == "mlp":
        mlp.main(args.input_sentence)
    elif args.classification_type == "lr":
        lrc.main(args.input_sentence)