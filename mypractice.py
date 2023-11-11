#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = 20
b = 20
print(id(a))
print(id(b))


# In[2]:


a=10
b=4
print(a % b)  # reminder


# In[3]:


a = int(input())
b = int(input())
c = a+b
print(c)


# In[ ]:


n = int(input())
#r = n%2
#is_even =(r ==0)
if n % 2 ==0:
#if is_even:
    print('n is even')
else:
    print('n is odd')


# In[ ]:


n = int(input())
m= int(input())
if n % 2 ==0:
    if m % 2 ==0:
        print('both even')
    else:
        print('n is even and m is odd')
else:
    print('n is odd')


# In[ ]:


# first n natural number
n = int(input())
c = 1
while c <=n:
    print(c)
    c = c+1


# In[ ]:


n = int(input())
a = 2
flag = False
while a<n:
    if (n%a==0):
        flag= True
    a = a+1
if flag:
    print('not prime')
else:
    print('prime')


# In[ ]:


# pattern
n= int(input())
i =1
while i<=n:
    j =1
    while j<=n:
        #print('*',end='')
        #print(i,end='')
        print(j,end=' ')
        j = j+1
    print()
    i = i+1


# In[ ]:


n=int(input())
i=1
while i<=n:
    j=1
    while j<=n:
        print(((n-j)+1),end='')
        j=j+1
    print()
    i=i+1


# In[ ]:


# star pattern
n=int(input())
i=1
while i<=n:
    j=1
    while j<=i:
        print('*',end=' ')
        #print(j,end=' ')
        j=j+1
    print()
    i=i+1


# In[ ]:


# for loop in range by default its start with 0 and stride is 1, range(start,stop,stride)
n= int(input())
#for i in range(n):
#for i in range(2,10,2):
for i in range(n,1,-1):
    print(i)


# In[ ]:


n = int(input())
for i in range(1,n+1,1):
    for a in range(n-i):
        print(' ',end=' ')
        
    for j in range(i,2*i,1):
        print('*',end=' ')
        
    for j in range(2*i-2,i-1,-1):
        print('*',end =" ")
    print()


# In[ ]:


#a =[i for i in range(20)]
a =[i**2 for i in range(20)]
print(a)


# In[8]:


s= int(input())
a= []
for i in range(s):
    ne= int(input())
    a.append(ne)
print(a)


# In[12]:


line = input()
l = line.split()
a= []
for s in l:
    a.append(int(s))
a


# In[14]:


l = input().split()
a = [int(s) for s in l]
a


# In[15]:


a = [int(s) for s in input().split()]
a


# In[24]:


a = [1,2,3,4,5,6]
#b=a[:1:-1]
a.reverse()
print(a)


# In[26]:


s = input().split()
m = int(s[0])
n = int(s[1])
l = []
for i in range(m):
    n_row= [int(i) for i in input().split()]
    l.append(n_row)


# In[27]:


l


# In[29]:


def fac(a):
    a_fac = 1
    for i in range(1,a+1):
        a_fac= a_fac*i
    return a_fac
fac(4)


# In[30]:


def swap(l):
    i=0
    while i+1 <len(l):
        l[i],l[i+1] = l[i+1],l[i]
        i = i+2
        
a = [1,2,3,4,5,6]
swap(a)
a
        


# In[35]:


def check_pass(password):
    s_char = '@,#,$,%,&,*,!,/,/,[]'
    if not any(char in s_char for char in password):
        return False
    if not any(char.isdigit() for char in password):
        return False
    if not any(char.islower for char in password):
        return False
    if not any(char.isupper for char in password):
        return False
    # if all requirement are met
    return True

user_pass = input('enter your password: ')

if check_pass(user_pass):
    print('right password')
else:
    print('try again')


# In[44]:


file = open('classes.txt','r')
#file_data = file.read()
#file_data = file.read(10)
file_data = file.readline()
print(type(file_data))
print(file_data)


# In[53]:


import csv
with open('compare.csv') as f:
    file = csv.reader(f)
    
    #for column in file:
        #print(column)
    file_list = list(file)
    print(file_list)
#print(type(file))
camera = []
for row in file_list[1:]:
    camera.append(row[2])
camera


# In[ ]:




