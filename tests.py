from fem1d import FEM1D
from fem2d import FEM2D
from mfem import MFEM
import numpy as np


####################################################### 1D Finite Elements ###########################################################
mesh_sizes = [2**i for i in range(4,11)]

#Trig 
#-u'' = f
trig2 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,1), coeffs = [lambda x: 0,lambda x: 0,lambda x: -1],f = lambda x: np.sin(x), u_p=lambda x: np.sin(x)) 
#-u'' + u = f
trig20 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,1), coeffs = [lambda x: 1,lambda x: 0,lambda x: -1],f = lambda x: 2*np.sin(x), u_p=lambda x: np.sin(x))
#-u'' + u' = f
trig21 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,1), coeffs = [lambda x: 0,lambda x: 1,lambda x: -1],f = lambda x: np.sin(x) + np.cos(x), u_p=lambda x: np.sin(x))
#-u'' + u' + u = f
trig210= FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,1), coeffs = [lambda x: 1,lambda x: 1,lambda x: -1],f = lambda x: 2*np.sin(x) + np.cos(x), u_p=lambda x: np.sin(x))
#(ku')' = f
ktrig2 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,1), coeffs = [lambda x: 0,lambda x: 0,lambda x: -x**2],f = lambda x: -2*x*np.cos(x) + (x**2)*np.sin(x), u_p=lambda x: np.sin(x))


#Polynomial
#-u'' = f
poly2 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,(1/8)*np.pi*(np.pi**2 - 4)), coeffs = [lambda x: 0,lambda x: 0,lambda x: -1],f = lambda x: -6*x, u_p=lambda x: x**3 - x) 
#-u'' + u = f
poly20 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,(1/8)*np.pi*(np.pi**2 - 4)), coeffs = [lambda x: 1,lambda x: 0,lambda x: -1],f = lambda x: x**3 - 7*x, u_p=lambda x: x**3 - x) 
#-u'' + u' = f
poly21 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,(1/8)*np.pi*(np.pi**2 - 4)), coeffs = [lambda x: 0,lambda x: 1,lambda x: -1],f = lambda x: 3*x**2 - 6*x -1 , u_p=lambda x: x**3 - x) 
#-u'' + u' + u = f
poly210 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,(1/8)*np.pi*(np.pi**2 - 4)), coeffs = [lambda x: 1,lambda x: 1,lambda x: -1],f = lambda x: x**3 + 3*x**2 - 7*x - 1 , u_p=lambda x: x**3 - x) 
#(ku')' = f
kpoly2 = FEM1D(N = 5,bounds = (0,np.pi/2), bc = (0,(np.pi/2)**3), coeffs = [lambda x: 0,lambda x: 0,lambda x: -np.exp(x)],f = lambda x: -3*(x**2)*np.exp(x) -6*x*np.exp(x), u_p=lambda x: x**3)

#Implementation
trials = [trig2,trig20,trig21,trig210,ktrig2,poly2,poly20,poly21,poly210,kpoly2]
for trial in trials:
    trial.get_errors(mesh_sizes, plot=False)
    #trial.plot_sol()


####################################################### 2D Finite Elements ###########################################################
mesh_sizes = [(n,n) for n in [8,16,32]]

#pure dirichlet example
bc = {'left':{'bc_type':'dirichlet','function': lambda x,y: np.sin(x+y)},
      'right':{'bc_type':'dirichlet','function': lambda x,y: np.sin(x+y)},
      'lower':{'bc_type':'dirichlet','function': lambda x,y: np.sin(x+y)},
      'upper':{'bc_type':'dirichlet','function': lambda x,y: np.sin(x+y)}}
fem = FEM2D(domain=(0,1,0,1),
            mesh_size=(5,5),
            coeffs=[1,1],
            f=lambda x,y: 3*np.sin(x+y),
            bc=bc, solution = lambda x,y: np.sin(x+y))
u_h = fem.solve()
fem.plot_sol()
fem.get_errors(mesh_sizes)


#mixed boundary example
bc = {'left':{'bc_type':'dirichlet','function': lambda x,y: y + np.sin(y)},
      'right':{'bc_type':'nuemann','function': lambda x,y: 2},
      'lower':{'bc_type':'nuemann','function': lambda x,y: -2},
      'upper':{'bc_type':'nuemann','function': lambda x,y: 1+np.cos(1)}}
fem = FEM2D(domain=(0,1,0,1),
            mesh_size=(10,10),
            coeffs=[0,1],
            f=lambda x,y: -2 + np.sin(y),
            bc=bc, solution = lambda x,y: x**2 + y + np.sin(y))
u_h = fem.solve()
fem.plot_sol()
fem.get_errors(mesh_sizes)

####################################################### Mixed Finite Elements ###########################################################

mesh_sizes = [(n,n) for n in [8,16,32]]

#example 1
f1 = lambda x,y:  2*np.sin(x+y) + y**2
f2 = lambda x,y:  - 2*np.sin(x+y) + 2*x*y
u1 = lambda x,y: np.sin(x+y)
u2 = lambda x,y: -np.sin(x+y)
p = lambda x,y: x*y**2 - 1/6

self = MFEM(bc=(0,1,0,1),numElements=(16,16),g=[u1,u2],f=[f1,f2],u=[u1,u2],p=p)
self.solve()
self.plot_sol()
#self.plot_mat()
self.plot_error_heat_map()
self.get_errors(mesh_sizes)

#example 2
f1 = lambda x,y: 24*y*(4-x**2)**2 - 32*(x**2)*y*(4-y**2) + 16*(4-x**2)*y*(4-y**2) + 300*x**2
f2 = lambda x,y: 32*x*(4-x**2)*(y**2) - 16*x*(4-x**2)*(4-y**2) - 24*x*(4-y**2)**2 + x*0
def u1(x,y): 
    return 4*((4 - x**2)**2)*y*(4 - y**2)
def u2(x,y): 
    return -4*x*(4 - x**2)*(4 - y**2)**2 
def p(x,y): 
    return 100*x**3

self = MFEM(bc=(-2,2,-2,2),numElements=(30,30),g=[u1,u2],f=[f1,f2],u=[u1,u2],p=p)
self.get_errors(mesh_sizes)  

#example 3
f1 = lambda x,y:  2*np.sin(x+y) + 2*np.cos(x+y) + y**2
f2 = lambda x,y:  -2*np.sin(x+y) -2*np.cos(x+y) + 2*x*y
u1 = lambda x,y: np.sin(x+y) + np.cos(x+y)
u2 = lambda x,y: -np.sin(x+y) - np.cos(x+y)
p = lambda x,y: x*y**2 - 1/6

self = MFEM(bc=(0,1,0,1),numElements=(64,64),g=[u1,u2],f=[f1,f2],u=[u1,u2],p=p)
#self.solve()
#self.plot_sol()
#self.plot_mat()
self.get_errors(mesh_sizes)
