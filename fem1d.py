import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy

class FEM1D:
    '''
    This class can solve up to 2nd order ODEs. 
    Uses peice wise linear finite elements. 
    
    N (number of intervals - controls granularity of the solution, will always default to equispaced mesh): int
    bounds (boundaries): tupple: (a,b): a<=x<=b
    bc (boundary conditions): tupple: (u_a,u_b) : u(a) = u_a, u(b) = u_b
    coeffs (ODE coefficients): list of length 3: 
        ex. 3u'' + 2u' + u = f --> [lambda x: 1,lambda x: 2,lambda x: 3] 
            -k(x)u'' - k'(x)u' = f --> [lambda x: 0,lambda x: -k'(x), lambda x: -k(x)]
            -6u'' = f --> [lambda x: 0,lambda x: 0,lambda x: -6]
    
    f (forcing function): lambda function: 
        ex. u'' = -6x --> f(x) = lambda x: -6*x
    '''
    def __init__(self,N,bounds,bc,coeffs,f,u_p = None):
        self.bounds = bounds
        self.a = bounds[0]
        self.b = bounds[1]
        self.bc = bc
        self.u_a = bc[0]
        self.u_b = bc[1]
        self.f = f
        self.N = N
        self.h = (self.b-self.a)/self.N
        self.mesh = np.linspace(self.a,self.b,num=self.N+1).tolist()
        self.coeffs = coeffs
        self.u_p = u_p
    
    @staticmethod
    def integrate(f,a,b):
        '''
        integrate the function f from bounds a to b

        Input:
            - a,b: integral bounds
            - f: lambda function

        Output:
            - yeild of integral of f from a to b
        '''
        #define quadrature (simpsons rule)
        return ((b-a)/6)*(f(a) + 4*f((a+b)/2) + f(b))
    
    def derivative(self,f,x):
        '''
        take the derivative of function f at point x
        Input:
            - f: lambda function
            - x: derivative will be evaluated at x
        Output:
            - yeild of derivative evaluated at point x 
        '''
        if x+2*self.h>self.b:
            #backward finite-difference
            return (3*f(x) - 4*f(x-self.h) + f(x-2*self.h))/(2*self.h)
        elif x-2*self.h<self.a:
            #forward finite_difference
            return (-f(x+2*self.h) + 4*f(x+self.h) - 3*f(x))/(2*self.h)
        else:
            #2nd order centerd finite-difference
            return (-f(x+2*self.h) + 8*f(x+self.h) - 8*f(x-self.h) + f(x-2*self.h))/(12*self.h)
        
        
    def pw_derivative(self,x):
        '''
        derivative of u_h, using the finite elements 
        Input:
            - x: derivative will be evaluated at x

        Output:
            - yeild of derivative evaluated at point x 
        '''
        for ix,i in enumerate(self.mesh):
            if x <= i:
                j = ix
                if j==0:
                    return self.u_h_vector[j]*(-1/self.h) + self.u_h_vector[j+1]*(1/self.h)
                else:
                    return self.u_h_vector[j-1]*(-1/self.h) + self.u_h_vector[j]*(1/self.h) 
              
    def phi(self,i,x):
        '''
        define the peice wise linear finite element / hat function
        Input:
            - i (int): index of the hat function
            - x (float): float input to the hat function 
        Output:
            - phi_i(x) = return value
        '''

        if (self.mesh[i-1]<x and self.mesh[i]>x):
            return (x-self.mesh[i-1])/self.h
        elif (self.mesh[i]<=x and self.mesh[i+1]>x):
            return (self.mesh[i+1]-x)/self.h
        else:
            return 0
    
    def matrix(self,m_type,coeff):
        '''
        Assmeble the Mass, Transport, or Stiffness Matrix based on input

        Input 
            - m_type: str :
                M : Mass Matrix --> for the 0th order component of the ODE
                T : Transport Matrix --> for the 1st order component of the ODE
                S : Stiffness Matrix --> for the 2nd order component of the ODE

            - coeffs (ODE coefficients): list of length 3: 
                ex. 3u'' + 2u' + u = f --> [lambda x: 1,lambda x: 2,lambda x: 3] 
                    -k(x)u'' - k'(x)u' = f --> [lambda x: 0,lambda x: -k'(x), lambda x: -k(x)]
                    -6u'' = f --> [lambda x: 0,lambda x: 0,lambda x: -6]
        Output:
            - Mass, Transport, or Stiffness Matrix
        '''
        
        #choose local matrix
        if m_type == 'T':
            local = lambda k,x1,x2: np.array([[(-1/(self.h))*self.integrate(lambda x: (k(x)*(x-x1))/self.h, x1, x2),
                                               (1/(self.h))*self.integrate(lambda x: (k(x)*(x-x1))/self.h, x1, x2)],
                                              [(-1/(self.h))*self.integrate(lambda x: (k(x)*(x2-x))/self.h, x1, x2),
                                               (1/(self.h))*self.integrate(lambda x: (k(x)*(x2-x))/self.h, x1, x2)]])
            
        elif m_type == 'M':
            local = lambda k,x1,x2: (1/(self.h**2))*np.array([[self.integrate(lambda x: k(x)*(x2-x)**2, x1, x2),
                                                               self.integrate(lambda x: k(x)*(x2-x)*(x-x1), x1, x2)],
                                                              [self.integrate(lambda x: k(x)*(x2-x)*(x-x1), x1, x2),
                                                               self.integrate(lambda x: k(x)*(x-x1)**2, x1, x2)]])
        elif m_type == 'S':
            local = lambda k,x1,x2: -1*self.integrate(k,x1,x2)*np.array([[1/(self.h**2),
                                                                          -1/(self.h**2)],
                                                                         [-1/(self.h**2),
                                                                          1/(self.h**2)]])
        else:
            raise('m_type options are M:Mass, S:Stiffness, T:Transport')
           
        #build global matrix 
        A = scipy.sparse.lil_matrix((self.N+1,self.N+1)) #We use sparse matrices to save memory
        for i in range(0,self.N):
            r,c = i,i
            local_w_coeff = local(coeff,self.mesh[i],self.mesh[i+1])
            A[r:r+local_w_coeff.shape[0], c:c+local_w_coeff.shape[1]] += local_w_coeff
        
        return A   
    
    def load_vector(self):
        '''
        generate load vector 
        the general problem form is A(u,w) = F(w)
        F(w) is the load vector

        Output:
            - Load Vector F(w)
        '''
        #Create Load Vector
        F = np.zeros(self.N+1)
        for i in range(1,self.N):
            F[i] = self.integrate(lambda x: self.f(x)*((x - self.mesh[i-1])/self.h), self.mesh[i-1], self.mesh[i]) + \
                   self.integrate(lambda x: self.f(x)*((self.mesh[i+1] - x)/self.h), self.mesh[i], self.mesh[i+1])
        F[0] = self.u_a
        F[-1] = self.u_b
        
        return F
    
    def solve(self):
        '''
        solve the ODE
        Output:
            - finite element solution for u_h
        '''
        F = self.load_vector()
        M = self.matrix('M',self.coeffs[0])
        T = self.matrix('T',self.coeffs[1])
        S = self.matrix('S',self.coeffs[2])
        A = M + T + S
        
        #Impose Boundary conditions
        A[0,0] = 1
        A[0,1] = 0
        A[-1,-1] = 1
        A[-1,-2] = 0
        
        u_h_vector = scipy.sparse.linalg.spsolve(A,F)
        self.u_h_vector = u_h_vector
        self.A = A
        return u_h_vector
    
    def u_h(self,x):  
        '''
        the finite element reconstruction of the solution u
        Input:
            - x: evaluate the reconstruction at dx
        Output:
            - u_h(x) = return value 
        '''

        #define the peicewise continous solution
        if x in self.mesh:
            return self.u_h_vector[self.mesh.index(x)]
        else:
            for ix,i in enumerate(self.mesh):
                if x < i:
                    j = ix
                    break
            return self.u_h_vector[j-1]*self.phi(j-1,x) + self.u_h_vector[j]*self.phi(j,x)     
    
    def plot_sol(self):
        '''
        Plot the finite element reconstruction of u(x) and the true solution
        '''

        if self.u_p!=None:
            mesh = np.linspace(self.a,self.b,num=1000).tolist()
            plt.plot(mesh,self.u_p(np.array(mesh)), 'b', label = 'True')
        plt.plot(mesh,[self.u_h(x) for x in mesh], 'r--', label = 'FEM')
        plt.legend()
        plt.show()

        return 


    def get_errors(self, mesh_sizes, plot = False):
        '''
        mesh_sizes: list of equal tuples [(n,n),(n+1,n+1)]
        plot: bool to plot the convergence behavior of u in h1 and l2

        Output:
            Convergence Rate Table
            return value: two lists, L2 errors for given meshes 
                                 and H1 errors for given meshes
        '''

        l2_errors = []
        h1_errors = []
        for N in mesh_sizes:
            self.__init__(N = N,bounds=self.bounds,bc=self.bc,coeffs = self.coeffs,f = self.f,u_p = self.u_p)
            self.solve()
            l2_err = np.sum([self.integrate(lambda x: (self.u_p(x) - self.u_h(x))**2,self.mesh[i],self.mesh[i+1]) for i in range(self.N)])
            h1_err = np.sum([self.integrate(lambda x: (self.u_p(x) - self.u_h(x))**2 + (self.derivative(self.u_p,x) - self.pw_derivative(x))**2,self.mesh[i],self.mesh[i+1]) for i in range(self.N)])
            l2_errors.append(l2_err**(1/2))
            h1_errors.append(h1_err**(1/2)) 

        h = np.array([abs(self.b-self.a)/N for N in mesh_sizes])
        h1_roc = np.polyfit(np.log10(h),np.log10(h1_errors),1)
        l2_roc = np.polyfit(np.log10(h),np.log10(l2_errors),1)

        print(f'''
             ____________________________________
            |Convergence Rates                   |
            |u(x,y) on H1 | {h1_roc[0]}                         
            |u(x,y) on L2 | {l2_roc[0]}                           
            |_____________|______________________|
            ''')  

        if plot:
            plt.loglog(h,h1_errors, color = 'gold', label= 'H1 Error')
            plt.loglog(h,l2_errors, color = 'g', label= 'L2 Error')
            plt.loglog(h,h, 'r--', label = 'Order 1')
            plt.loglog(h,h**2, 'b', linestyle='dashed', label = 'Order 2')
            plt.legend()
            plt.show()   

        return l2_errors,h1_errors


if __name__ == '__main__':
    ###EXAMPLES####

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
    mesh_sizes = [2**i for i in range(4,11)]
    trials = [trig2,trig20,trig21,trig210,ktrig2,poly2,poly20,poly21,poly210,kpoly2]

    for trial in trials:
        trial.get_errors(mesh_sizes, plot=True)
        trial.plot_mat()
        trial.plot_sol()
