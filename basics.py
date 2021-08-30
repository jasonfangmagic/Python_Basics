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

#string method

