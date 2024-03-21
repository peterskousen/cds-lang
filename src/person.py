class Person:
    species = "Homo sapiens"
    def __init__(self, name, likes):
        self.name = name
        self.likes = likes

    def hello(self):
        print("Hello, " + self.name)

    def preferences(self):
        print("I like " + self.likes + "!")
