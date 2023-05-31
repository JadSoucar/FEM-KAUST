from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import fem_utils


class MFEM:
    def __init__(self,bc,numElements,f,g,u,p):
        '''
        solve mixed finite element problems with the following structure
        -laplacian(u) + gradient(p) = f
                             div(u) = 0
                                  u = g

        where integral(p) = 0
        using a P1 element uniform mesh for p and a red refinment of the first mesh for u

        **f,g,u,p are all defined within the context of the above problem structure**

        bc: tuple (a,b,c,d)
            the x axis will span a to b
            the y axis will span c to d 
            ex. (0,1,0,1) yeilds the unit square

        numElements: tuple (nx,ny) containing number of elements on the width 
                     and height of the rectangular domain defined by the bc paramter


        '''

        #Construct P1-iso-P2 meshes
        self.bc = bc
        self.a,self.b,self.c,self.d = bc
        self.numElements = numElements
        self.nx,self.ny = self.numElements
        dx_hat,dy_hat = 1/self.nx,1/self.ny
        self.tt3,xt3_hat,yt3_hat,self.tt6,xt6_hat,yt6_hat,self.corse_to_fine = fem_utils.mesh_t3_iso_t6(self.nx,self.ny,dx_hat,dy_hat)
        self.t3numNodes, self.t6numNodes = len(xt3_hat),len(xt6_hat)

        #map unit domain to omega
        x_transform = lambda x_hat: self.a + (self.b-self.a)*x_hat
        y_transform = lambda y_hat: self.a + (self.d-self.a)*y_hat
        self.xt3 = np.array([x_transform(x_hat) for x_hat in xt3_hat])
        self.yt3 = np.array([y_transform(y_hat) for y_hat in yt3_hat])
        self.xt6 = np.array([x_transform(x_hat) for x_hat in xt6_hat])
        self.yt6 = np.array([y_transform(y_hat) for y_hat in yt6_hat])
        self.dx = abs((self.b-self.a)/self.nx)
        self.dy = abs((self.c-self.b)/self.ny)
        self.lowert3,self.rightt3,self.uppert3,self.leftt3,self.all_boundaries_t3 = fem_utils.get_boundaries(self.t3numNodes,self.nx,self.ny)
        self.lowert6,self.rightt6,self.uppert6,self.leftt6,self.all_boundaries_t6 = fem_utils.get_boundaries(self.t6numNodes,2*self.nx,2*self.ny)

        #forcing equation
        self.f = f
        self.f1,self.f2 = f
        #boundary 
        self.g = g
        self.g1,self.g2 = g

        #Truth 
        self.u = u 
        self.u1,self.u2 = u
        self.p = p 

        #Solution Vector 
        self.sol = 0 #placeholder 
        

    #################################################### Mixed FEM Assembly ######################################################################
        
        
    def solve(self):
        '''
        solve mixed finite element problems with the following structure
        -laplacian(u) + gradient(p) = f
                             div(u) = 0
                                  u = g

        where integral(p) = 0

        using a P1 element mesh for p and a red refinment of the first mesh for u

        Output:
            - the solution vector the Mixed FEM scheme: [u1,u2,p,x]
                where u1 is a t6numNodes sized component of sol
                      u2 is a t6numNodes sized component of sol
                      p is a t3numNodes sized component of sol
                      x is a scalar value representing the lagrange multiplier for the
                        mean valued condition

        '''

        #Nodal Functions
        N_grad_ref = [np.array([-1,-1]),
                      np.array([1,0]),
                      np.array([0,1])]

        self.A11 = fem_utils.grad_u_grad_u(self.t6numNodes,self.tt6,self.xt6,self.yt6,self.dx/2,self.dy/2)
        print('Stiffness Matrix Constructed')  

        #Construction B11
        self.B11,self.B22 = fem_utils.grad_u_v(self.t6numNodes,self.t3numNodes,self.tt6,self.tt3,self.xt6,self.yt6,
                                                self.xt3,self.yt3,self.dx/2,self.dx/2,self.dx,self.dy,self.corse_to_fine)

        print('B Matrix Constructed')

        #Add Mean Value Constraint 
        M = np.zeros(self.t3numNodes)
        for K3 in self.tt3:
            k3_coords,k3_node_order = fem_utils.get_physical_coords(K3,self.xt3,self.yt3)
            for node in k3_node_order:
                M[node] += (1/6)*self.dx*self.dy

        print('Mean Value Contraint Imposed')

        #Construct S
        zero_t6 = scipy.sparse.lil_matrix((self.t6numNodes,self.t6numNodes))
        zero_t3 = scipy.sparse.lil_matrix((self.t3numNodes,self.t3numNodes))
        zero_vec_t6 = np.zeros(self.t6numNodes).reshape(self.t6numNodes,1)
        zero_vec_t3 = np.zeros(self.t3numNodes).reshape(self.t3numNodes,1)
        single_zero = np.zeros(1).reshape(1,1)
        M = M.reshape(self.t3numNodes,1)
        self.S = scipy.sparse.bmat([[self.A11,zero_t6,-1*self.B11.T,zero_vec_t6,],
                                   [zero_t6,self.A11,-1*self.B22.T,zero_vec_t6],
                                   [-1*self.B11,-1*self.B22,zero_t3,M],
                                   [zero_vec_t6.T,zero_vec_t6.T,M.T,single_zero]]).tolil()


        print('S Matrix Assembled')
        #Construct F
        mass = fem_utils.u_u(self.t6numNodes,self.tt6,self.xt6,self.yt6)
        F1 = mass@np.array([self.f1(self.xt6[ix],self.yt6[ix]) for ix in range(self.t6numNodes)])
        F2 = mass@np.array([self.f2(self.xt6[ix],self.yt6[ix]) for ix in range(self.t6numNodes)])


        #Impose Dirchlet conditions 
        for ix in self.all_boundaries_t6:
            zero_row1 = np.zeros(2*self.t6numNodes + self.t3numNodes + 1)
            zero_row1[ix] = 1
            zero_row2 = np.zeros(2*self.t6numNodes + self.t3numNodes + 1)
            zero_row2[ix+self.t6numNodes] = 1
            self.S[ix] = zero_row1
            self.S[ix+self.t6numNodes] = zero_row2


        for ix in self.all_boundaries_t6:
            F1[ix] = self.g1(self.xt6[ix],self.yt6[ix])
            F2[ix] = self.g2(self.xt6[ix],self.yt6[ix])


        #Construct L
        zeros_vector_t3 = np.zeros(self.t3numNodes+1)
        L = np.hstack((F1,F2,zeros_vector_t3))

        print('Load Vector Assembled')

        #Solve System
        sol = scipy.sparse.linalg.spsolve(self.S.tocsr(),L)

        print('S@x = L Solved')
        
        self.sol = sol
        
        return True
    

    #################################################### Mixed FEM Solution Plotting #############################################################
    def plot_sol(self):
        '''
        plot solutions from the solution vector yeilded by the solve() function
        plots generated will be 
            the true and FEM U1
            the true and FEM U2
            the true and FEM P
        '''
        if type(self.sol) == int:
            raise('Must Call slove() before ploting solution')


        #create mesh for truth 
        true_u1 = [self.u1(self.xt6[ix],self.yt6[ix]) for ix in range(len(self.xt6))]
        u1_true_mesh = np.array([true_u1[n:n+2*self.nx+1] for n in range(0,len(self.xt6),2*self.nx+1)])
        true_u2 = [self.u2(self.xt6[ix],self.yt6[ix]) for ix in range(len(self.xt6))]
        u2_true_mesh = np.array([true_u2[n:n+2*self.nx+1] for n in range(0,len(self.xt6),2*self.nx+1)])
        true_p = [self.p(self.xt3[ix],self.yt3[ix]) for ix in range(len(self.xt3))]
        p_true_mesh = np.array([true_p[n:n+self.nx+1] for n in range(0,len(self.xt3),self.nx+1)])

        #create FEM meshes
        xt6_mesh = np.array([self.xt6[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        yt6_mesh = np.array([self.yt6[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        xt3_mesh = np.array([self.xt3[n:n+self.nx+1] for n in range(0,self.t3numNodes,self.nx+1)])
        yt3_mesh = np.array([self.yt3[n:n+self.nx+1] for n in range(0,self.t3numNodes,self.nx+1)])

        u_h1_mesh = np.array([self.sol[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        u_h2_mesh = np.array([self.sol[n:n+2*self.nx+1] for n in range(self.t6numNodes,2*self.t6numNodes,2*self.nx+1)]) 
        p_mesh = np.array([self.sol[n:n+self.nx+1] for n in range(2*self.t6numNodes,len(self.sol)-1,self.nx+1)]) 

        #plot u1
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(xt6_mesh, yt6_mesh, u1_true_mesh, color="blue",alpha=.75)
        ax.set_title('U1 True')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.plot_surface(xt6_mesh, yt6_mesh, u_h1_mesh, color="red",alpha=.75)
        ax1.set_title('U1 FEM')
        plt.show()

        #plot u2
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(xt6_mesh, yt6_mesh, u2_true_mesh, color="blue",alpha=.75)
        ax.set_title('U2 True')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.plot_surface(xt6_mesh, yt6_mesh, u_h2_mesh, color="red",alpha=.75)
        ax1.set_title('U2 FEM')
        plt.show()

        #plot p
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(xt3_mesh, yt3_mesh, p_true_mesh, color="blue",alpha=.75)
        ax.set_title('P True')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.plot_surface(xt3_mesh, yt3_mesh, p_mesh, color="red",alpha=.75)
        ax1.set_title('P FEM')
        plt.show()

        return True

    
    def plot_mat(self):
        '''
        plot the stiffnes Matrices A,B,S 
            where A is the matrix belonging to the laplacian(u) term
            where B is the matrix belonging to the gradient(p) term
            where S is the assembled matrix S = [[A,B.T],
                                                 [B,0]]
        '''

        cmap = ListedColormap(['w', 'w', 'r'])
        B11 = self.B11.copy()
        xnon0, ynon0 = B11.nonzero()
        for ix,iy in zip(xnon0,ynon0):
            B11[ix,iy] = 1
        plt.matshow(B11.toarray(), interpolation='nearest',cmap=cmap)
        plt.show()

        A11 = self.A11.copy()
        xnon0, ynon0 = A11.nonzero()
        for ix,iy in zip(xnon0,ynon0):
            A11[ix,iy] = 1
        plt.matshow(A11.toarray(), interpolation='nearest',cmap=cmap)
        plt.show()

        S = self.S.copy()
        xnon0, ynon0 = S.nonzero()
        for ix,iy in zip(xnon0,ynon0):
            S[ix,iy] = 1
        plt.matshow(S.toarray(), interpolation='nearest',cmap=cmap)
        plt.show()
        

    #################################################### Mixed FEM Error Calculations ######################################################################
    def partial(self,f_hat,x_hat,y_hat,wrt):
            '''
            take the partial derivative of function f at a point x or y
            f: lambda function
            wrt: 'x','y', derivative taken with respect to (wrt) x or y
            x: float
            y: float
            '''
            if wrt == 'x':
                h = self.dx/2
                lower_bound = self.a
                upper_bound = self.b
                z = x_hat
                f = lambda x: f_hat(x,y_hat)
            elif wrt == 'y':
                h = self.dy/2
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



    def errors(self):
        '''
        generates the error i.e ||sol - truth||_L2 and ||sol-truth||_H1
        i.e. the error term for the FEM sol over the mesh slected 

        Output:
            - a list of size (3,) containing in the first position 
            ||p_h - p||_L2, second ||u_h - u||_L2, third ||u_h-u||_H1

        '''
        u1_h = self.sol[:self.t6numNodes]
        u2_h = self.sol[self.t6numNodes:2*self.t6numNodes]
        p_h = self.sol[2*self.t6numNodes:-1]

        u1_l2_err = 0
        u2_l2_err = 0
        u_x_semi_err = 0
        u_y_semi_err = 0
        for K6 in tqdm(self.tt6):
            coords,nodes = fem_utils.get_physical_coords(K6,self.xt6,self.yt6)
            u1_l2_err += (1/3)*(1/2)*(self.dx/2)*(self.dy/2)*sum([(u1_h[node] - self.u1(coord[0],coord[1]))**2 for coord,node in zip(coords,nodes)])
            u2_l2_err += (1/3)*(1/2)*(self.dx/2)*(self.dy/2)*sum([(u2_h[node] - self.u2(coord[0],coord[1]))**2 for coord,node in zip(coords,nodes)])

            grad_K6 = [fem_utils.nodal_grad(np.array([-1,-1]),coords),fem_utils.nodal_grad(np.array([1,0]),coords),fem_utils.nodal_grad(np.array([0,1]),coords)]
            grad_x_u_h,grad_y_u_h = np.array(grad_K6).T
            u1_h_K6 = np.array([u1_h[nodes[0]], u1_h[nodes[1]], u1_h[nodes[2]]])
            u2_h_K6 = np.array([u2_h[nodes[0]], u2_h[nodes[1]], u2_h[nodes[2]]])
            grad_u = [[self.partial(self.u1,coord[0],coord[1],'x'),self.partial(self.u2,coord[0],coord[1],'y')] for coord,node in zip(coords,nodes)]
            grad_x_u, grad_y_u = np.array(grad_u).T
            u_x_semi_err += (1/3)*(1/2)*(self.dx/2)*(self.dy/2)*np.sum((u1_h_K6@grad_x_u_h - grad_x_u)**2)
            u_y_semi_err += (1/3)*(1/2)*(self.dx/2)*(self.dy/2)*np.sum((u2_h_K6@grad_y_u_h - grad_y_u)**2)

        p_l2_err = 0
        for K3 in tqdm(self.tt3):
            coords, nodes = fem_utils.get_physical_coords(K3,self.xt3,self.yt3)
            p_l2_err += (1/3)*(1/2)*(self.dx)*(self.dy)*sum([(p_h[node] - self.p(coord[0],coord[1]))**2 for coord,node in zip(coords,nodes)])
            

        p_l2_err = np.sqrt(p_l2_err)
        u_l2_err = np.sqrt(u1_l2_err + u2_l2_err)
        u_h1_err = np.sqrt(u1_l2_err + u2_l2_err + u_x_semi_err + u_y_semi_err)

        return p_l2_err,u_l2_err,u_h1_err


    def get_errors(self,mesh_sizes,plot=False):
        '''
        plot errors across a list of mesh sizes 

        Input 
            - mesh_sizes: list of equal tuples [(n,n),(n+1,n+1),...]
            - plot: bool to plot the convergence behaviors of u and p in h1 and l2
        
        Output
            - errors: (3,len(mesh_sizes)) size matrix of errors for u1,u2,p
                       for example errors[1,1] corresponds to the error for u1 using 
                       the first mesh size in mesh_sizes
        '''
        errors = []
        for mesh_size in tqdm(mesh_sizes):
            self.__init__(self.bc,mesh_size,self.f,self.g,self.u,self.p)
            self.solve()
            errors.append(self.errors())
        
        p_errors,u_l2_errors,u_h1_errors = np.array(errors).T
        h = np.array([1/min(mesh_size) for mesh_size in mesh_sizes])
        p_l2_roc = np.polyfit(np.log10(h),np.log10(p_errors),1)
        u_l2_roc = np.polyfit(np.log10(h),np.log10(u_l2_errors),1)
        u_h1_roc = np.polyfit(np.log10(h),np.log10(u_h1_errors),1)

        print(f'''
             _________________________________
            |Convergence Rates                |
            |u(x,y) on H1 | {u_h1_roc[0]}|               
            |u(x,y) on L2 | {u_l2_roc[0]}|    
            |p(x,y) on L2 | {p_l2_roc[0]}|             
            |_____________|___________________|
     
            ''') 

        if plot:
            plt.loglog(h,u_h1_errors, color = 'gold', label= 'U H1 Error')
            plt.loglog(h,u_l2_errors, color = 'g', label= 'U L2 Error')
            plt.loglog(h,p_errors, color = 'pink', label= 'P L2 Error')
            plt.loglog(h,h, 'r--', label = 'Order 1')
            plt.loglog(h,h**2, 'b', linestyle='dashed', label = 'Order 2')
            plt.legend()
            plt.show()     

        return np.array(errors).T 


    def plot_error_heat_map(self):
        '''
        plot error heat maps for u1, u2, and p
        error in this case refers simply to the absolute error betweeen the 
        FEM solution and the True Solution in very element
        '''
        if type(self.sol) == int:
            raise('Must Call slove() before ploting solution')

        #create mesh for truth 
        self.true_u1 = [self.u1(self.xt6[ix],self.yt6[ix]) for ix in range(self.t6numNodes)]
        self.true_u2 = [self.u2(self.xt6[ix],self.yt6[ix]) for ix in range(self.t6numNodes)]
        self.true_p = [self.p(self.xt3[ix],self.yt3[ix]) for ix in range(self.t3numNodes)]
        
        #create FEM meshes
        xt6_mesh = np.array([self.xt6[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        yt6_mesh = np.array([self.yt6[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        xt3_mesh = np.array([self.xt3[n:n+self.nx+1] for n in range(0,self.t3numNodes,self.nx+1)])
        yt3_mesh = np.array([self.yt3[n:n+self.nx+1] for n in range(0,self.t3numNodes,self.nx+1)])
        self.u_h1_mesh = np.array([self.sol[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        self.u_h2_mesh = np.array([self.sol[n:n+2*self.nx+1] for n in range(self.t6numNodes,2*self.t6numNodes,2*self.nx+1)]) 
        self.p_mesh = np.array([self.sol[n:n+self.nx+1] for n in range(2*self.t6numNodes,len(self.sol)-1,self.nx+1)]) 
        
        #plot u1       
        a = self.u_h1_mesh -  np.array([self.true_u1[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        fig, ax = plt.subplots()
        im = ax.imshow(a, cmap='hot')
        cbar = ax.figure.colorbar(im)
        ax.set_title('U1 Error')
        plt.show()

        #plot u2
        a = self.u_h2_mesh -  np.array([self.true_u2[n:n+2*self.nx+1] for n in range(0,self.t6numNodes,2*self.nx+1)])
        fig, ax = plt.subplots()
        im = ax.imshow(a, cmap='hot')
        cbar = ax.figure.colorbar(im)
        ax.set_title('U2 Error')
        plt.show()

        #plot p
        a = self.p_mesh -  np.array([self.true_p[n:n+self.nx+1] for n in range(0,self.t3numNodes,self.nx+1)])
        fig, ax = plt.subplots()
        im = ax.imshow(a, cmap='hot')
        cbar = ax.figure.colorbar(im)
        ax.set_title('P Error')
        plt.show()
        
        return True