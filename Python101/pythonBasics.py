#        Start
# age = 20
# age = 30
# price = 19.95
# first_name = "Abdi"
# print(str(4) + " years old")  # 20 + `years old` doesn't work
# print(4 * " years old") # this works:  years old years old years old years old
# print("age : ", age, " :price : ", price)
# print("name : " + first_name)
# first = 'abdi'
# last = 'mustefa'
# print(f'{first} {last} is the full name') # f : formatter
# ---------------------------------------------------------------
#           input
# ---------------------------------------------------------------
# last_name = input("ur last name please ? ")
# print("Hi " + first_name + "  " +last_name)
# ---------------------------------------------------------------
#        types int, str, float
# ---------------------------------------------------------------
# birth_year = input("Birth year.")  # input function return string
# age = 2024 - int(birth_year)  # cast it to int type
# print("age is : ", age)  # can't concatenate int to str
# first = 10.1
# second = 11
# print("sum is :" + str(first + second))
# ---------------------------------------------------------------
#           strings :  strings are objects & immutables
# ---------------------------------------------------------------
# course = "Python for Beginner"
# print(course.upper())  # PYTHON FOR BEGINNER
# print(course.find('y'))  # 1
# print(course.find('for'))  # 7
# print("Python" in course)  # True

# ---------------------------------------------------------------
#        arithmetic expression
# ---------------------------------------------------------------
# print(10 / 3)  # float -> 3.3333
# print(10 // 3)  # int  -> 2
# print(10 % 3)   # 1
# print(10 ** 3)  # exponent operator -> 1000
# x = 10
# x += 3  # 13
# ---------------------------------------------------------------
#           comparison operators
# ---------------------------------------------------------------
# print(10 > 5)  # True
# print(not 10 > 5)  # False

# ---------------------------------------------------------------
#          if/else
# ---------------------------------------------------------------
# temp = 30
# if temp > 30:
#     print("It's a hot day")
#     print("Turn on AC")
# elif temp < 30:
#     print("It's a cold day")
# else:
#     print("You good")
# print("Done")
# ---------------------------------------------------------------
#          while
# ---------------------------------------------------------------
# i = 1
# while i <= 5:
#     print(i)
#     print(i * "Hello : ")
#     i += 1
# ---------------------------------------------------------------
#           list : [ ]
# ---------------------------------------------------------------
# names = ["John", "Bob", "Sara", "Mosh"]
# print("Bob" not in names)
# names[0] = "Jon"
# print(names[0])  # Jon
# print(names[-1])  # Mosh
# print(names[0:2])  # Jon, Bob, Sara
# names.append("Tom")
# print(names)
# names.insert(0, "Abdi")
# names.insert(1, 99)
# print(names)
# print("Abdi" in names)
# print(len(names))
# print(names.count("Abdi"))
# ---------------------------------------------------------------
#           for loop / while loop
# ---------------------------------------------------------------
# numbs = [1, 2, 3, 4, 5]
# for numb in numbs:
#     print(numb)
# i = 0
# while i < len(numbs):
#     print(numbs[i])
#     i += 1
# ---------------------------------------------------------------
#           range
# ---------------------------------------------------------------
# numbers = range(5, 10)  # 5, 6, 7 , 8, 9
# numbers = range(5, 10, 2)  # 5, 7, 9
# for i in numbers:
#     print(i)
# ---------------------------------------------------------------
#           tuples : ( )
# ---------------------------------------------------------------
# numbers = (1, 2, 3, 4)  # no append, remove, clear, pop . . . .
# numbers[0] = 1   # ERROR: tuple object can't be update
# ---------------------------------------------------------------
#            Unpacking : works for both List and Tuples
# ---------------------------------------------------------------
# point = (1, 2, 3)
# x, y, z = point  # same as x = point(0), y = point(1) , z = point(2)
# ---------------------------------------------------------------
#           2D list : [ [ ], [] ]
# ---------------------------------------------------------------
# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(matrix[0])  # [1,2,3]
# print(matrix[0][1])  # 2
# for row in matrix:
#     for item in row:
#         print(item)
# ---------------------------------------------------------------
#          Dictionaries
# ---------------------------------------------------------------
customer = {
    "name": "Abdi",
    "Age": 30,
    "is_passed": True,
    1: 5
}
print(customer["name"])  # throw exception if key not found
print(customer[1])
print(customer.get("nameNN"))  # return null if key not found
print(customer.get("nameNN", "new name"))  # get for key 'nameNN' of default value 'new name'
customer[2] = 98
print(customer)  # {'name': 'Abdi', 'Age': 30, 'is_passed': True, 1: 5, 2: 98}

