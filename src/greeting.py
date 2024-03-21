import argparse
from person import Person

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", 
                        "-n",
                        type = str,
                        required = True
    )
    parser.add_argument("--likes",
                        "-l"
    )
    return parser.parse_args()


def main():
    args = argument_parser()
    person = Person(args.name, args.likes)
    person.hello()
    person.preferences()

if __name__ == "__main__":
    main()