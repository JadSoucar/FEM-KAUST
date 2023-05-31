import numpy as np
import scipy

################################################# MESH ########################################################
def mesh_t3(nx,ny,delta_x,delta_y):

		'''
		Creation of a mesh for the unit square [0,1]^2.
		This is used to create meshes for P1 elements.

		Input:
			- nx, ny: number of elements on x-axis and y-axis
			- delta_x, delta_y: size of elements on x-axis and y-axis

		Output:
			- (topo, x, y): mesh with parameters given in input
		'''

		topo = np.zeros((2*nx*ny,3),dtype=int)
		x = np.linspace(0,nx*delta_x,nx+1)
		y = np.linspace(0,ny*delta_y,ny+1)
		(x,y) = np.meshgrid(x,y)
		x = np.concatenate(x)
		y = np.concatenate(y)
		i_el = 0
		for i in range(ny):
			for j in range(nx):
				nodes = np.array([ i    * (nx + 1 ) + j,
								  (i+1) * (nx + 1 ) + j + 1,
								  (i+1) * (nx + 1 ) + j])
				topo[i_el][:] = nodes
				i_el = i_el+1
				nodes = np.array([ i * (nx + 1) + j,
								   i * (nx + 1) + j + 1,
								(i+1)* (nx + 1) + j + 1])
				topo[i_el][:] = nodes
				i_el = i_el+1

		return topo, x, y


def mesh_t3_iso_t6(nx,ny,delta_x,delta_y):

	'''
	Creation of two embedded meshes for the unit square [0,1]^2.
	This is used to create meshes ofr P1-iso-P2 elements.

	Input:
		- nx, ny: number of elements on x-axis and y-axis
		- delta_x, delta_y: size of elements on x-axis and y-axis

	Output:
		- (tt3,xt3,yt3)=(topo,x,y): coarse mesh with parameters given in input
		- (tt6,xt6,yt6)=(topo,x,y): fine mesh (double number of elements and size halved)
		- corse_to_fine: connection between the two meshes (number of pressure el x 4)
	'''

	(tt3,xt3,yt3) = mesh_t3(nx,ny,delta_x,delta_y)

	(tt6,xt6,yt6) = mesh_t3(2*nx,2*ny,delta_x/2,delta_y/2)

	corse_to_fine = np.zeros((2*nx*ny,4),dtype=int)

	for i in range(ny):
		for j in range(nx):
			lr_coarse = (2*nx)*i+2*j+1
			fine = np.zeros((1,4))
			fine[0][0] = (j*4)+8*nx*i+1
			fine[0][1] = (j*4)+8*nx*i+2
			fine[0][2] = (j*4)+8*nx*i+3
			fine[0][3] = (j*4)+8*nx*i+4*nx+3
			corse_to_fine[lr_coarse] = fine
			ul_coarse = (2*nx)*i+2*j
			fine[0][0] = (j*4)+8*nx*i
			fine[0][1] = (j*4)+8*nx*i+4*nx
			fine[0][2] = (j*4)+8*nx*i+4*nx+1
			fine[0][3] = (j*4)+8*nx*i+4*nx+2
			corse_to_fine[ul_coarse] = fine
	return tt3,xt3,yt3,tt6,xt6,yt6,corse_to_fine


def get_boundaries(numNodes,nx,ny):
	'''
	Generates the global node number coresponding to each of the 4 boundaries of 
	the rectangular domain defined by parameters a,b,c,d
	Input:
			- nx, ny: number of elements on x-axis and y-axis
			- numNodes: total number of nodes in the domain 
	Output:
		- lower,right,upper,left: 4 seperate arrays of global node numbers for the lower,
								  right, upper, and left boundaries
		- all_boundaries: a concatenation of the lower,right,upper, and left vectors

	'''
	lower = np.arange(nx+1) 
	right = np.arange(nx,numNodes,ny+1)
	upper = np.flip(np.arange(numNodes-1,numNodes-2-nx,-1))
	left = np.arange(0,numNodes-ny,ny+1)
	all_boundaries = set(np.concatenate([lower,right,upper,left]))
	return lower,right,upper,left,all_boundaries




################################################# Physical_Reference_Transformations ########################################################


def get_B(physical_coords):
	'''
	Let F(x_hat) = B@x_hat + b be a transformation from the reference 
	element to the physical element. 
	This is used to create the Transformation. 
	Input:
		- physical coords: ((x1,y1),(x2,y2),(x3,y3)) <-- the (x,y) coordinates
							of the physical P1 element 

	Output:
		- the B matrix and b scalar vector that make up the F transformation
			  B: (2x2)
			  b: (2,) 

	'''

	p1,p2,p3 = physical_coords
	x1,y1 = p1
	x2,y2 = p2
	x3,y3 = p3
	B = np.array([[x2-x1,x3-x1],
				  [y2-y1,y3-y1]])
	scalar = np.array([x1,y1])

	return B,scalar

def get_physical_coords(K,x,y):
	'''
	This function converts a set of global node numbers to their respective (x,y)
	coordinates in the physical domain 

	Input:
		- K: list of three global node number that make up a P1 element
		- x: the x mesh produced by mesh_t3_iso_t6 
		- y: the y mesh produced by mesh_t3_iso_t6

	Output:
		- physical coords: ((x1,y1),(x2,y2),(x3,y3)) <-- the (x,y) coordinates
							of the physical P1 element 
		- node_order: the global node numbers in K rearanged to ensure that 
					  the corner containing ht 90 degree angle is first in the list
	'''

	if K[1]<K[2]:
		'''
		 /|
		/_|
		'''
		p2,p1,p3 = K
		physical_coords = [(x[p1],y[p1]),(x[p2],y[p2]),(x[p3],y[p3])]
		node_order = [p1,p2,p3]

	else:
		'''
		 __
		| /
		|/
		'''
		p2,p3,p1 = K
		physical_coords = [(x[p1],y[p1]),(x[p2],y[p2]),(x[p3],y[p3])]
		node_order = [p1,p2,p3]

	return physical_coords, node_order



def nodal_grad(grad_ref_N,physical_coords):
	'''
	Let N_alpha be the nodal function for each node in the P1 element 
	This function computes the laplacian(N_alpha)

	Input:
		- grad_ref_N: a (2,) vector representing the gradient of the nodal function 
					  N_alpha in the reference domain
		- physical coords: ((x1,y1),(x2,y2),(x3,y3)) <-- the (x,y) coordinates
							of the physical P1 element 

	Output:
		- a (2,) vector representing the gradient of the nodal function 
		  N_alpha in the physical domain

	'''
	B,scalar = get_B(physical_coords)
	return np.linalg.inv(B.T)@grad_ref_N



############################################### ASSEMBLY #####################################################

def grad_u_grad_u(numNodes,topo,xmesh,ymesh,dx,dy):
	'''
	generate stiffness matrix of form
	 _                                                              _
	| (grad u_0, grad_u_0) (grad u_1, grad_u_0) (grad u_2, grad_u_0) |
	| (grad u_0, grad_u_1)                              :            |
	| (grad u_0, grad_u_2)                              :            |
	|          :                                        :            |  = A
	|          :                                                     | 
	| (grad u_0, grad_u_numNodes)  (grad u_numNodes, grad_u_numNodes)|         
	|_                                                              _|

	Input:
		- numnodes: number of degrees of freedom 
		*the following are produced by the mesh functions 
		- topo: list with size (numNodes,(3,1)) containing nodes in each element of mesh
		- xmesh: list of x coords in mesh 
		- ymesh: list of y coords in mesh
		- dx,dy: width and height of each element respectivly

	Output:
		- Stiffness matrix size (numNodes,numNodes)
	'''
	N_grad_ref = [np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
	A = scipy.sparse.lil_matrix((numNodes,numNodes))
	for ix,K in enumerate(topo):
		#get triangle coordinates
		physical_coords, node_order = get_physical_coords(K,xmesh,ymesh)
		#calculate local A11
		d_x = np.array([nodal_grad(N_grad_ref[i],physical_coords) for i in range(3)])
		local = (.5)*(dx)*(dy)*(d_x@d_x.T)
		#populate global A11
		for i,p1 in enumerate(node_order):
			for j,p2 in enumerate(node_order):
				A[p1,p2] += local[i,j]  

	return A



def grad_u_v(u_numNodes,v_numNodes,u_topo,v_topo,u_xmesh,u_ymesh,v_xmesh,v_ymesh,u_dx,u_dy,v_dx,v_dy,corse_to_fine):
	'''
	generate two grad_u_v matrices of the following form. 
	One matrix selects the x element of the grad, and other selects y 
	 _                                                                _
	| (grad u_0, v_0)       (grad u_1, v_0)      (grad u_2, v_0)       |
	| (grad u_0, v_1)                                   :              |
	| (grad u_0, v_2)                                   :              |
	|          :                                        :              | = B
	|          :                                                       | 
	| (grad u_0, v_{v_numNodes})  (grad u_{u_numNodes}, v_{v_numNodes})|         
	|_                                                                _|

	Input:
		- u_numnodes: number of degrees of freedom for the u mesh 
		*the following are produced by the mesh functions 
		- u_topo: list with size (numNodes,(3,1)) containing nodes in each element of the u mesh
		- u_xmesh: list of x coords in the u mesh 
		- v_ymesh: list of y coords in the umesh
		- u_dx,u_dy: width and height of each u element respectivly

		**same for v, but each argument belongs to the v mesh

	Output:
		- B11 matrix of size (v_numNodes,u_numNodes) where grad_u selects the x element
		- B22 matrix of size (v_numNodes,u_numNodes) where grad_u selects the y element

	'''
	N_grad_ref = [np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
	B11 = scipy.sparse.lil_matrix((v_numNodes,u_numNodes))
	B22 = scipy.sparse.lil_matrix((v_numNodes,u_numNodes))
	for i, K3 in enumerate(v_topo):
		K3_physical_coords, K3_node_order = get_physical_coords(K3,v_xmesh,v_ymesh)
		for K3_node,K3_coord in zip(K3_node_order,K3_physical_coords):
			sub_triangles = corse_to_fine[i]
			for K6 in sub_triangles:
				K6_physical_coords, K6_node_order = get_physical_coords(u_topo[K6],u_xmesh,u_ymesh)

				if K3_coord in K6_physical_coords:
					K6_integral = ((1/6)*(u_dx)*(u_dy))*(1/2 + 1/2 + 1)
				elif sum([1 if k6_node in K3_physical_coords else 0 for k6_node in K6_physical_coords])==1:
					K6_integral =  ((1/6)*(u_dx)*(u_dy))*(1/2)
				elif sum([1 if k6_node in K3_physical_coords else 0 for k6_node in K6_physical_coords])==0:
					K6_integral =  ((1/6)*(u_dx)*(u_dy))*(1/2 + 1/2)

				for j,K6_node in enumerate(K6_node_order):
					grad_N = nodal_grad(N_grad_ref[j],K6_physical_coords)
					B11[K3_node,K6_node] += K6_integral*grad_N[0]
					B22[K3_node,K6_node] += K6_integral*grad_N[1]

	return B11, B22


def u_u(numNodes,topo,xmesh,ymesh):
	'''
	generate mass matrix of form
	 _                                          _
	| (u_0, u_0) (u_1, u_0) (u_2, u_0) 			 |
	| (u_0, u_1)                :                |
	| (u_0, u_2)                :                |
	|          :                :                |  
	|          :                                 | 
	| (u_0, u_numNodes)  (u_numNodes, u_numNodes)|         
	|_                                          _|

	Input:
		- numnodes: number of degrees of freedom 
		*the following are produced by the mesh functions 
		- topo: list with size (numNodes,(3,1)) containing nodes in each element of mesh
		- xmesh: list of x coords in mesh 
		- ymesh: list of y coords in mesh

	Output:
		- Mass matrix size (numNodes,numNodes)
	'''
	#calculate Mass Matrix
	M = scipy.sparse.lil_matrix((numNodes,numNodes))
	for K in topo:
		#calculate local
		physical_coords, node_order = get_physical_coords(K,xmesh,ymesh)
		B,scalar = get_B(physical_coords)
		det = abs(np.linalg.det(B))
		local = (det/24)*np.array([[2,1,1],[1,2,1],[1,1,2]])
		#populate global
		for i,p1 in enumerate(node_order):
			for j,p2 in enumerate(node_order):
				M[p1,p2] += local[i,j]

	return M