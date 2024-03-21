import argparse

class Person:
    species = "Homo sapiens"
    def __init__(self, name):
        self.name = name

    def hello(self):
        print("Hello, " + self.name)

    def preferences(self):
        print("I like Python!")




def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", "-n",
        type = str
    )
    return parser.parse_args()




def main():
    person = Person(argument_parser().name)
    person.hello()
    person.preferences()

if __name__ == "__main__":
    main()