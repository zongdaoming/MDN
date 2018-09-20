def f(a,*args):
    print(args)
    print(a)

# 表示任意数目参数收集
def func(a,*args):
    print(args)
# func(1,2,3,4,5)

# **允许将关键字参数转化为字典
def func2(**kargs):
    print(kargs)
# func2(a=1,b=2)

def func3(a,*pargs,**kargs):
    print(a,pargs,kargs)
# func3(1,2,3,x=3,y=4)

def func4(a,b,c,d):
    print(a,b,c,d)
# args=(1,2,3,4)
# func4(*args)

def func5(a,b,c,d):
    print(a,b,c,d)
# kargs = {'a':1,'b':2,'c':3,'d':4}
# func5(**kargs)


def func6(a,b,c,d,e,f):
    print(a,b,c,d,e,f)

# args = (2,3)
# kargs = {'d':4,'e':5}
#
# func6(1,*args,f=6,**kargs)

def tracer(func7,*args,**kargs):
    print('calling:',func7.__name__)
    return func7(*args,**kargs)

def func7(a,b,c,d):
    return a+b+c+d
# print(tracer(func7,1,2,c=3,d=4))


def fun8(a,b,c,d,e):
    print(a,b,c,d,e)
# args=(1,2,3,4,5)
# fun8(*args)