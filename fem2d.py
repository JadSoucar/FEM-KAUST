import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy
from tqdm import tqdm
import fem_utils

#Create Mesh 
class FEM2D:
    def __init__(self,domain,mesh_size,coeffs,f,bc,solution):
        '''
        domain: tuple (a,b,c,d)
            the x axis will span a to b
            the y axis will span c to d 
            ex. (0,1,0,1) yeilds the unit square
        mesh_size: tuple (l,h)
            how many triangles on each side
            ex. (100,100) 100 triangles on the x axis, and y axis
        f (forcing function/surface tension): lambda function 
            ex. laplatian(u) + u = f
        bc (boundary conditions): dictionary 
            note that the domain will always be rectangular, so 
            there are 4 sides where the boundary conditions need to be determined
            (this is most basic case, this class is not equiped for boundary conditions
            that change within a side of the domain)\
            {('left'|'right'|'upper'|'lower'): {bc_type:('nuemann'|'dirichlet'), 
                                                function: lambda function}
        solution (truth values): lambda function
        coeffs: list
            ex. -laplacian(u) + u = f --> [1,1]
                -laplacian(u) = f --> [0,1]
                u = f --> [1,0]
                laplacian(u) = f --> [0,-1]
                laplacian(u) - u = f --> [-1,-1]

        '''
        #Mesh Prep
        self.domain = domain
        self.a,self.b,self.c,self.d = self.domain
        self.nx, self.ny = mesh_size
        self.numNodes = (self.nx+1)*(self.ny+1)
        self.dx = abs((self.b-self.a)/self.nx)
        self.dy = abs((self.c-self.b)/self.ny)
        self.topo, self.x, self.y = fem_utils.mesh_t3(self.nx,self.ny,self.dx,self.dy)
        self.all_nodes = [(self.x[ix],self.y[ix]) for ix in range(self.numNodes)]
        lower,right,upper,left,all_boundaries = fem_utils.get_boundaries(self.numNodes,self.nx,self.ny)

        #boundary condition prep 
        self.f = f
        self.bc = bc
        self.dirichlet = []
        self.nuemann = []
        for side,nodes in zip(['lower','right','upper','left'],[lower,right,upper,left]):
            if bc[side]['bc_type']=='dirichlet':
                self.dirichlet.append([nodes,bc[side]['function']])
            elif bc[side]['bc_type']=='nuemann':
                self.nuemann.append([nodes,bc[side]['function']])
            else:
                raise('boundary conditions can be nuemann or dirichlet')


        self.coeffs = coeffs
        self.u_p = solution 


    def partial(self,f_hat,x_hat,y_hat,wrt):
        '''
        take the partial derivative of function f at a point x or y

        Input:
            - f: lambda function 
            - wrt: 'x','y', derivative taken with respect to (wrt) x or y
            - x: float
            - y: float

        Output:
            - Partial derivative withe respect to "wrt" evaluated at (x,y)
        '''
        if wrt == 'x':
            h = self.dx
            lower_bound = self.a
            upper_bound = self.b
            z = x_hat
            f = lambda x: f_hat(x,y_hat)
        elif wrt == 'y':
            h = self.dy
            lower_bound = self.c
            upper_bound = self.d
            z = y_hat
            f = lambda y: f_hat(x_hat,y)
        else:
            raise('options for wrt are "x" or "y"')
            
        if z+2*h>lower_bound:
            #backward finite-difference
            return (3*f(z) - 4*f(z-h) + f(z-2*h))/(2*h)
        elif z-2*h<upper_bound:
            #forward finite_difference
            return (-f(z+2*h) + 4*f(z+h) - 3*f(z))/(2*h)
        else:
            #2nd order centerd finite-difference
            return (-f(z+2*h) + 8*f(z+h) - 8*f(z-h) + f(z-2*h))/(12*h)

    
    def solve(self):
        '''
        solve finite element problems with the following structure
        -a*laplacian(u) + b*u = f
        using a P1 element mesh 

        Output:
            - the solution vector, i.e the finite element solution of u
        '''
        #calculate stiffness matrix
        S = fem_utils.grad_u_grad_u(self.numNodes,self.topo,self.x,self.y,self.dx,self.dy)
        print('Stiffness Matrix Constructed')
                            
        #calculate mass matrix
        M = fem_utils.u_u(self.numNodes,self.topo,self.x,self.y)
        print('Mass Matrix Constructed')
                    
        #compile stiffness and mass
        A = (self.coeffs[1]*S + self.coeffs[0]*M).tolil()
        print('LHS Assembled')

        #Calculate Load Vector
        L = M@np.array([self.f(self.x[ix],self.y[ix]) for ix in range(self.numNodes)])

        #Calculate Neuman vector
        N = np.zeros(self.numNodes)
        for nodes,func in self.nuemann:
            for ix in range(len(nodes[:-1])):
                x1,y1 = self.x[nodes[ix]],self.y[nodes[ix]]
                x2,y2 = self.x[nodes[ix+1]],self.y[nodes[ix+1]]
                N[nodes[ix]] += (func(x1,y1)+func(x2,y2))*(self.dx/4)
                N[nodes[ix+1]] += (func(x1,y1)+func(x2,y2))*(self.dx/4)

        #add neuman vector to the load vector 
        L = L+N
        print('RHS Assembled')

        #Impose Dirichlet Condition on Stiffness/Mass
        for nodes,func in self.dirichlet:
            for ix in nodes:
                sparse_row = np.zeros(self.numNodes)
                sparse_row[ix] = 1
                A[ix] = sparse_row

        #impose direchlet condition on load vector
        for nodes,func in self.dirichlet:
            for ix in nodes:
                L[ix] = func(self.x[ix],self.y[ix]) 

        #get finite element solution vector
        u_h = scipy.sparse.linalg.spsolve(A.tocsr(),L)
        print('A@x = L Solved')

        self.u_h = u_h
        return u_h
    
    def plot_sol(self): 
        '''
        plot solutions from the solution vector yeilded by the solve() function
        plots generated will be 
            the true and FEM U 
        '''

        #get truth
        true = np.array([self.u_p(self.x[ix],self.y[ix]) for ix in range(self.numNodes)])
        
        #plot results
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        x_mesh = np.array([self.x[n:n+self.nx+1] for n in range(0,self.numNodes,self.nx+1)])
        y_mesh = np.array([self.y[n:n+self.nx+1] for n in range(0,self.numNodes,self.nx+1)])
        u_h_mesh = np.array([self.u_h[n:n+self.nx+1] for n in range(0,self.numNodes,self.nx+1)])
        true_mesh = np.array([true[n:n+self.nx+1] for n in range(0,self.numNodes,self.nx+1)])
        ax.plot_surface(x_mesh, y_mesh, u_h_mesh, color="red", alpha = .75)

        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.plot_surface(x_mesh, y_mesh, true_mesh, color="blue",alpha=.75)

        plt.show()

        
    def error(self):
        '''
        generates the error i.e ||sol - truth||_L2 and ||sol-truth||_H1
        i.e. the error term for the FEM sol over the mesh slected 

        Output:
            - a list of size (2,) containing in the first position 
            ||sol - truth||_H1 and in the second ||sol - truth||_L2

        '''
        errors = []
        for K in self.topo:
            #get triangle coordinates
            physical_coords, node_order = fem_utils.get_physical_coords(K,self.x,self.y)

            #compile integrand for H1
            grad_ref_N1 = np.array([-1,-1])
            grad_ref_N2 = np.array([1,0])
            grad_ref_N3 = np.array([0,1])
            grad_N1 = fem_utils.nodal_grad(grad_ref_N1,physical_coords)
            grad_N2 = fem_utils.nodal_grad(grad_ref_N2,physical_coords)
            grad_N3 = fem_utils.nodal_grad(grad_ref_N3,physical_coords)
            grad_K = self.u_h[node_order[0]]*grad_N1 \
                   + self.u_h[node_order[1]]*grad_N2 \
                   + self.u_h[node_order[2]]*grad_N3
                    
            integrand_h1 = lambda x,y: (grad_K[0]-(self.partial(self.u_p,x,y,'x')))**2 \
                                   +(grad_K[1]-(self.partial(self.u_p,x,y,'y')))**2 \
                                   +(self.u_h[self.all_nodes.index((x,y))]-self.u_p(x,y))**2
            
            #compile integrand for L2
            integrand_l2 = lambda x,y: (self.u_h[self.all_nodes.index((x,y))]-self.u_p(x,y))**2
            

            #calculate error
            x0,y0 = physical_coords[0][0],physical_coords[0][1]
            x1,y1 = physical_coords[1][0],physical_coords[1][1]
            x2,y2 = physical_coords[2][0],physical_coords[2][1]
            errors.append([(self.dx*self.dy*(1/6))*(integrand_h1(x0,y0) + integrand_h1(x1,y1) + integrand_h1(x2,y2)),
                           (self.dx*self.dy*(1/6))*(integrand_l2(x0,y0) + integrand_l2(x1,y1) + integrand_l2(x2,y2))])
        
        errors = np.array(errors)
        return [np.sum(errors.T[0])**(1/2),np.sum(errors.T[1])**(1/2)]
        
        
    def get_errors(self,mesh_sizes,plot=False):
        '''
        mesh_sizes: list of equal tuples [(n,n),(n+1,n+1)]
        plot: bool to plot the convergence behavior of u in h1 and l2

        Output:
            Convergence Rate Table
            return value: two lists, L2 errors for given meshes 
                                 and H1 errors for given meshes
        '''
        errors = []
        for mesh_size in tqdm(mesh_sizes):
            self.__init__(self.domain,mesh_size,self.coeffs,self.f,self.bc,self.u_p)
            self.solve()
            errors.append(self.error())
        
        
        h = np.array([1/min(mesh_size) for mesh_size in mesh_sizes])
        h1_errors, l2_errors = np.array(errors).T
        h1_roc = np.polyfit(np.log10(h),np.log10(h1_errors),1)
        l2_roc = np.polyfit(np.log10(h),np.log10(l2_errors),1)

        print(f'''
             _________________________________
            |Convergence Rates                |
            |u(x,y) on H1 | {h1_roc[0]}|               
            |u(x,y) on L2 | {l2_roc[0]}|                 
            |_____________|___________________|
     
            ''')  

        if plot:
            plt.loglog(h,h1_errors, color = 'gold', label= 'H1 Error')
            plt.loglog(h,l2_errors, color = 'g', label= 'L2 Error')
            plt.loglog(h,h, 'r--', label = 'Order 1')
            plt.loglog(h,h**2, 'b', linestyle='dashed', label = 'Order 2')
            plt.legend()
            plt.show()   

        return l2_errors, h1_errors


if __name__ == '__main__':
    mesh_sizes = [(n,n) for n in [8,16,32,64,128]]


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


   