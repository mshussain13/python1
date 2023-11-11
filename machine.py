#!/usr/bin/env python
# coding: utf-8

# In[1]:


string = 'this is a room'
string.upper()


# In[2]:


len(string)


# In[3]:


string.replace('this','that')


# In[4]:


string.count('a')


# In[5]:


str = 'this is programming'
str.find('is')


# In[6]:


str.find('programming')


# In[7]:


fruits = 'i like mangoes,apple,banana,kivi'
fruits.split(',')


# In[8]:


s = 'india is a secular democratic repulic and all humen being live together'
s.split('a')


# In[11]:


tup = (100,'amir',True,3.13)
tup[-1]


# In[12]:


tup[:]


# In[13]:


tup[1:3]


# In[17]:


tup[1] = 'amir'


# In[16]:


tup1 = (1,'subh')
tup+tup1


# In[18]:


tup1*4


# In[19]:


min(tup)


# In[20]:


t =(1,4,6,9)
min(t)


# In[21]:


max(t)


# In[30]:


l =[1,'b',2,3,8,'n']
l.reverse()
l


# In[23]:


l


# In[24]:


l.append('sam')


# In[31]:


l.reverse()
l


# In[36]:


l1= ['sam','ram','subh','aam']
l1.insert(1,'rama')
l1
l1.sort()
l1


# In[39]:


d = {'a':1,'b':2,'sam':43,'er':67}
d.values()


# In[42]:


d['m']=10
d['sam']= 23
d


# In[43]:


d.pop('sam')
d


# In[44]:


d1 = {'w':78,'df':89,}
d.update(d1)
d


# In[46]:


age = 18
if age >17:
    print('you can not vote')
else:
    print('you vote')


# In[54]:


a = 8
b = 10
c = 15
if a >b & a<c:
    print('a is greatest')
elif b>a & b>c:
    print('b is greatest')
else:
    print('c is greatest')

    


# In[55]:


l1 = ['a','b','c']
if l1[1] == 'b':
    l1[1] = 'd'
l1


# In[56]:


i = 1
while i<= 10:
    print(i)
    i+= 1


# In[58]:


i =1
n=5
while i<=10:
    #print(n,' * ',i,' = ',n*i)
    print(n*i)
    i+= 1


# In[62]:


l1 = [1,2,3,4,5]
i = 0
while i < len(l1):
    l1[i] = l1[i]+200
    i= i+1
l1


# In[63]:


l1 = ['balck','white','yelo','cum']
l2 = ['yolo1','yolo2','yolo3','yolo4']
for i in l1:
    for j in l2:
        print(i,j)


# In[65]:


def add(x):
    return x+2
add(2)


# In[67]:


def hell():
    print('hello world')
hell()


# In[69]:


# lambda function
g = lambda x: x*x
g(3)


# In[72]:


l1 = [87,89,80,40,60,76,33]
#l2 = list(filter(lambda x: (x%2 == 0),l1))  # lambda with filter
l3 = list(map(lambda x: x*3,l1)) # lambda with map
l3


# In[88]:


import numpy as np
#n = np.array([21,20,34,45,65])
n = np.array([[23,12,123,44],[87,80,67,23]]) # multi
print(n.shape) # (2,4) arrsy shape
n.shape = (4,2)  # change array
print(n.shape)   # (4,2)
print(type(n))
n


# In[79]:


import numpy as np
#n1 = np.zeros((2,3))
n = np.ones((5,5))
n


# In[80]:


import numpy as np
#initializing numpy array with some number
n = np.full((3,3),2)
n


# In[81]:


import numpy as np
n = np.arange(1,50,5)
n


# In[85]:


import numpy as np
#with random numbers
n = np.random.randint(1,1000,10)
n


# In[92]:


import numpy as np
n = np.array([1,2,3])
n1 = np.array([5,6,7])
#np.vstack((n,n1))

#np.hstack((n,n1))

np.column_stack((n,n1))


# In[103]:


import numpy as np
n = np.array([1,23,45,56])
n1 = np.array([12,45,67,78])

#np.sum([n+n1])
#np.sum([n+n1],axis = 0)  # for column by column sum
#np.sum([n+n1],axis = 1)  # for row by row sum

np.mean(n1)
#np.median(n)
#np.std(n1)


# In[109]:


# save and load
import numpy as np
n = np.array([1,23,45,56])
np.save('m_numpy',n)
n2 = np.load('m_numpy.npy')
print(n2)


# In[111]:


name = 'ms'
age = 27
print(f'i am {name} and i am {age} years old' )


# In[113]:


print(','.join(['sam','man','ran','john']))


# In[114]:


phrase = 'there is a lion in jungle who torture every animal of jungle'

sc = {}

for letter in phrase:
    if letter in sc:
        sc[letter] +=1
    else:
        sc[letter] = 1
print(sc)


# In[117]:


import pandas as pd
#p = pd.Series([1,3,4,5,7])
p = pd.Series([1,3,4,5,7],index = ['a','b','c','d','e']) # to change series index
p
#print(type(p))


# In[125]:


import pandas as pd
pd.DataFrame({'name':['sam','subh','ran','kalp'],'age':[12,45,56,14]})



# In[138]:


import pandas as pd

compare = pd.read_csv('/home/shussain/Downloads/compare.csv')
#print(compare)
compare.head()    # first 5 row

compare.tail()  # last 5 row

compare.describe()  #describe information of csv

compare.shape      # show row and column of csv file


# In[143]:


import pandas as pd
com = pd.read_csv('/home/shussain/Downloads/compare.csv')

#com.iloc[2:5,:4]  # give result about perticular row and column

com.loc[0:4,('Camera','vtLpPredicted')]  # give perticuler column name


# In[148]:


import pandas as pd
com = pd.read_csv('/home/shussain/Downloads/compare.csv')

com.drop('Camera',axis = 1)  # for droping column

com.drop(2,axis = 0)   # for droping row value


# In[152]:


import pandas as pd
com = pd.read_csv('/home/shussain/Downloads/compare.csv')
com.mean()

com.median()

com.min()

com.max()


# In[157]:


import pandas as pd
com = pd.read_csv('/home/shussain/Downloads/compare.csv')

#com['LpValidity'].value_counts()  # counts value of perticuler column

com.sort_values(by = 'vtLpConfidence')  # sort values


# In[161]:


import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1,11)
y = 2*x

#plt.plot(x,y)
plt.plot(x,y,color = 'r',linestyle = ':')
plt.title('line')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()


# In[162]:


import numpy as np
from matplotlib import pyplot as plt

x = [10,20,30,40,50,60,70,80]
y = [1,2,3,4,5,6,7,8]

plt.scatter(x,y)

plt.show()



# In[165]:


ages = {'sam':23, 'ram':34, 'jam':12}
for pair in ages.items():
    print(pair)


# In[168]:


def fre_di(data):
    freq = {}
    for elem in data:
        if elem in freq:
            freq[elem] += 1
        else:
            freq[elem] = 1
    return freq

#fre_di('this is myy room')
fre_di([1,2,3,4,3,2,3,2,4,4,4,5])


# In[173]:


def my_fun(name,place):
    print(f'hello im {name} and i live in {place}')
my_fun('ms','pune')


# In[176]:


def total_cals(bill,tip = 10):
    total = bill*(1+0.01*tip)
    total = round(total,2)
    print(f'Please pay ${total}')
#total_cals(150)
total_cals(300,20)


# In[178]:


def area(l,b):
    return l*b
cube = area(20,10)
print(f'area of cube is: {cube}')


# In[180]:


def cube(side):
    v = side **3
    s = 6 * (side**2)
    return v,s
values = cube(4)
print(values)

# unpack and stre values in two diffenrt variables
volume , area = cube(8)
print(f'volume of cube is {volume} and surface area of cube is {area} !')


# In[182]:


def my_sum(*args):
    sum = 0
    for arg in args:
        sum +=arg
    return sum
sum = my_sum(10,20,3,34,45,67)
print(sum)


# In[ ]:




