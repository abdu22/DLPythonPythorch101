class Mammal:
    def walk(self):
        print("walk")


class Dog(Mammal):
    def bark(self):
        print("bark")


class Cat(Mammal):
    def miaw(self):
        print("miaw")


dog1 = Dog()
dog1.walk()
dog1.bark()

cat = Cat()
dog1.walk()
cat.miaw()