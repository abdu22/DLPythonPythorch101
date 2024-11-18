# 1234 --> print One Two Three Four

def converter(number_input):
    number_dict = {"1": "One", "2": "Two","3": "Three","4": "Four", "5":"Five", "6":"Six", "7":"Seven", "8":"Eight", "9": "Nine"}
    ans = ""
    for char in number_input:
        try:
            ans += number_dict.get(char) + " "
        except TypeError:
            print("Invalid value: " + char)

    return ans


number_input = input("Give your Number : ")
print(converter(number_input))

# for cr in number:12
#     print(cr)