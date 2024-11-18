def greet_user(name, last_name):
    print(f"Hi there, {name} {last_name}!")


print("Start")
greet_user("abdi", "mustefa")  # Positional argument
greet_user(last_name="mustefa", name="abdi")  # Key word argument
greet_user("abdi", last_name="mustefa")  # if both, key word argument comes after positional
print("End")


def square(number):
    return number * number

print(square(2))