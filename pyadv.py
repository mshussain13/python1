
'''
string = 'how are you doing'
string1 = string.split()
print(string1)

new_string = ' '.join(string1)
print(new_string)


var = 3.1234
#string = 'the varible is {}'.format(var)
string = f'the variable is {var}'
print(string)



add10 = lambda x: x+10
print(add10(2))

mult = lambda x,y: x+y
print(mult(2,7))


points = [(1,2),(13,2),(4,-1),(10,4)]
#points_sorted = sorted(points)  # sorting respect to x axis 
points_sorted = sorted(points, key = lambda x: x[1]) # sorting to y axis
print(points_sorted)


a = [1,2,3,4,5]

b = map(lambda x: x*2,a)
print(list(b))


try:
    a= 5/1
    b= a+ 10
except ZeroDivisionError as e:
    print(e)
except TypeError as e:
    print(e)
else:
    print('everything is fine')

finally:
    print('cleaning up..')



class ValueTooHighError(Exception):
    pass
    
def test_value(x):
    if x > 100:
        raise ValueTooHighError('value is too high')
try:
    test_value(200)
except ValueTooHighError as e:
    print(e)
'''
# string formatting

#name =  'tom'
#print("%s is a lowyer" %name)

#num = 10
#print("tom id no is: %d" %num)

'''
list = []
for i in range(10):
    list.append(i**2)
print(list)

# list comperhencian basicaly sorting the list code

alist = [i**2  for i in range(10)]
print(alist)
'''
# opps hard coded
class Rectangle:
    w = 10
    h= 20
    
    def area(self):
        return self.w * self.h
        
rect = Rectangle()
print(rect.area())

# dynamic
class Rectangle:
    def __init__(self,h,w): # constructure
        self.h = h
        self.w = w
        
    def area(self):
        return self.w * self.h
        
rect = Rectangle(10,18)
print(rect.h)
print(rect.area())



