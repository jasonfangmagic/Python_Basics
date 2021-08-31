#Input
print("Hello what's your name")
name = input()
print("Hello,", name)

#operator
num1 = 34
num2 = 3

#only have integer
print(num1 // num2)

#only residual
print(num1 % num2)

#expotion
print(num1 ** num2)


#convert numbers

print("pick a number")

num1 = input()

print("pick another number")

num2 = input()

sum = int(num1) + int(num2)

print("Your Number is", sum)


print(2 == 4)

age = input("Input your age; ")

if int(age) == 15:
    print("Hey, your age is 15")

else: print("You're not 15")


#Chained Conditional and Nested if statement
x =3
y = 9
if not (y == x or x +y == 6):
    print("run")

else:
    print(":(")


if x == 3:
    if y == 4:
        print("x = 3, y = 4")
    else:
        print("x = 2, y != 4")
else:
    print("x != 3")


#for loop

for x in range(0, 10): #start, stop, step
    print(x)

loop= True

while loop:
    name = input("insert sth: ")
    if name == "stop":
        break



#Lists and Tuples

fruits = ["apple", "pear", 3]

print(fruits[1])

fruits.append("orange")

print(fruits)

fruits[1] = "blueberry"

print(fruits)

position = (2, 3, 4)


# For Loop iteration


for i in fruits:
    if i == "apple":
        print(i)
    else:
        print("not apple")

#string methods

text = input("input sth: ")
print(text.strip())
print(len(text))

print(text.lower())

text = input("input sth: ")
print(text.split("."))

#Slice Operator

fruits = ["apple", "pear", 3]
text = "Hello I am Jason"
print(text[6:])

#steps
print(text[1:1])

fruits[1:1] = ["Blueberries"]
print(fruits)
#insert

#functions

def addtwo(x):
    return x+2

x=5
y=addtwo(x)
print(y)

def writestring(x):
    return print(x)

writestring("hello")

file = open("basics.txt", "r")
f = file.readlines()

print(f)


#using .count and .find
string1 = 'hello'

print(string1.find('o'))
print(string1.count('z'))

#optional parameters

def func(x, text='2'):
    print(x)
    if text == '1':
        print('text is 1')
    else:
        print('text is not 1')

func('jason')

#try and except
#global and local variables

newVar = 9
loop = True

def func(x):
    global newVar
    newVar = 7
    if x == 6:
        return newVar
    print(newVar)

func(6)

#objects and classes

x = 'string'
print(type(x))

class number():
    def __init__(self, num):
        self.var = num

    def display (self, x):
        print(x)

num = number()
print(23)

num.display(num.var)