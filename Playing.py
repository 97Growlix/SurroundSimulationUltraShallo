import matplotlib.pyplot as plt
import numpy as np

def CalcRfSpider(X):
    Rf = 5 + abs(X)*10/40

    return Rf

X = np.linspace(-45,45,22)

#RfSpider = [CalcRfSpider(u) for u in X]

#print(RfSpider)


#plt.plot(X, RfSpider)
#plt.show()

#print(np.argwhere(45, X))

Arr1 = [(1.2,5), (3.3333, 4.132)]
Arr2 = np.array([0.5, 0.1])

result= [
    (a*s, b*s) 
    for (a,b),s in zip(Arr1, Arr2)
]
print(result)

