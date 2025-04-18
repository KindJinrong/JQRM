'''
utf-8;
author;
We should use physics unit rad, m, .
For a new 6-DOF parallel, .
'''
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device ='cpu'
dtype= torch.float64
pi = torch.tensor(torch.pi,dtype=dtype)
'''
forward kinematics
given rotation matrix $^{\mathscr{K}_{\mathscr{K}_{A}}R$ and, position vector $\mathbf{r}$，(-pi,pi]
@ = torch.matmul() support broadcat, torch.mm not support
torch.linalg.cross $\time$ denotes the multiplication symbol, 
the fork product to find the vertical vector only holds in 3D and 7D spaces.
'''
class Parallel_backward():
    def __init__(self,a,b,l_1,l_2,theta):
        '''
        vector: r
        matrix: R
        value: a,b,l_1,l_2
        theta = pi/6
        i denotes the ith sub-chain, i = 0,1,2 and, j= 0,1
        '''
        # self.r = r.to(device)
        # self.R = R.to(device)
        self.a = a.to(device)
        self.b = b.to(device)
        self.l_1 = l_1.to(device)
        self.l_2 = l_2.to(device)
        self.theta = theta.to(device) # rad between the two sub-chains
        self.theta_chain = (2*pi-3*self.theta)/6
    def screw_matrix_z(self,alpha_z):
        screw_matrix_1 = torch.cos(alpha_z)
        screw_matrix_2 = torch.sin(alpha_z)
        screw_matrix = torch.tensor([[screw_matrix_1,-screw_matrix_2,0],
                                     [screw_matrix_2,screw_matrix_1,0],
                                     [0,0,1.0]],dtype = self.a.dtype,device = device)
        return screw_matrix
    def gene_vector_a(self,i):
        # i =0,1,2
        vector_a = torch.tensor([self.a,0,0],dtype = self.a.dtype,device = device)
        vector_a = self.screw_matrix_z(i*2*pi/3)@vector_a
        return vector_a
    
    def gene_vector_b(self,j,i):
        #  j=0,1,i=0,1,2
        vector_b = torch.tensor([self.b,0,0],dtype = self.a.dtype,device = device)
        vector_b = self.screw_matrix_z((-1)**(j+1)*self.theta_chain+i*2*pi/3)@vector_b         
        # print('vector_b',vector_b)
        return vector_b
    
    def _backward_solver(self,j,i):
        # j = 0,1,i=0,1,2
        s_1 = torch.tensor([0,1,0], dtype = self.a.dtype,device = device)
        s_1 = self.screw_matrix_z(i*2*pi/3)@s_1
        s_2_m = self.r-self.gene_vector_b(j,i)+self.R@self.gene_vector_a(i)
        s_2 = torch.linalg.cross(s_1,s_2_m)
        s_2 = s_2/torch.linalg.vector_norm(s_2,ord=2)
        s_3 = torch.tensor([0,1,0],dtype = self.a.dtype, device = device)
        s_3 = self.screw_matrix_z(i*2*pi/3)@s_3
        s_3 = self.R@s_3 
        d = torch.linalg.cross(s_2,s_3)
        d = d/torch.linalg.vector_norm(d,ord=2)        
        c = torch.linalg.cross(d,s_2)
        w = s_2_m + (-1)**(j+1)*self.l_1*c-self.l_2*d
        q = torch.linalg.vector_norm(w,ord=2)
        return q,w
        
    def bacward_solver(self,r,R):
        self.R = R.to(device) 
        self.r = r.to(device)       
        q_array = torch.empty((3,2),dtype = self.a.dtype,device= device)
        for i in range(3):
            q,_ = self._backward_solver(0,i)
            q_array[i,0] = q
            q,_ = self._backward_solver(1,i)
            q_array[i,1] = q
        return q_array.view(6)

    def gen_matrix_r_c(self):
        r_c_array = torch.empty(18,dtype = self.a.dtype,device= device)
        for i in range(3):
            for j in range(2):
                b = self.gene_vector_b(j,i)
                _,w = self._backward_solver(j,i)
                r_c_array[i*6+j*3:i*6+j*3+3] = b+w
        return r_c_array    
'''
Inverse motion, closed-loop equation + geometric constraint solution
--------------
Forward kinematics, unconstrained optimization problems, Newton's method for superlinear approximation
'''      
class Parallel_forward(Parallel_backward):
    def __init__(self,a,b,l_1,l_2,array_q,theta):
        
        super().__init__(a,b,l_1,l_2,theta)
        self.array_q = array_q.to(device)
        
    def gene_vector_r_a(self,r_1,r_2,i):
        # i = 0,1,2,r_2 = r_c,2,i
        s_1 = torch.tensor([0,1,0], dtype = self.a.dtype,device = device) 
        s_1 = self.screw_matrix_z(i*2*pi/3)@s_1
        b_1 = self.gene_vector_b(0,i)
        s_2 = torch.linalg.cross(s_1,(r_1-b_1))
        d = torch.linalg.cross(s_2,r_2-r_1)
        d_norm = d/torch.linalg.vector_norm(d,ord=2)     
        r_a = (r_1+r_2)/2+self.l_2*d_norm
        return r_a,d,s_2
    
    
    def nonlinear_eqs(self,r_all_vector):
        '''
        array_q_{6,},r_00_{3,}
        ''' 
        r_all_vector = r_all_vector.to(device)
        array_q = self.array_q
        r_00 = r_all_vector[:3]
        r_10 = r_all_vector[3:6]
        r_01 = r_all_vector[6:9]
        r_11 = r_all_vector[9:12]
        r_02 = r_all_vector[12:15]
        r_12 = r_all_vector[15:18]
        eq_1 = torch.sum((r_10-r_00)**2)-4*self.l_1**2
        eq_2 = torch.sum((r_11-r_01)**2) -4*self.l_1**2
        eq_3 = torch.sum((r_12-r_02)**2) -4*self.l_1**2 
        #
        eq_4 = torch.sum((r_00-self.gene_vector_b(0,0))**2) -array_q[0]**2
        eq_5 = torch.sum((r_10-self.gene_vector_b(1,0))**2)-array_q[1]**2
        eq_6 = torch.sum((r_01-self.gene_vector_b(0,1))**2)-array_q[2]**2
        eq_7 = torch.sum((r_11-self.gene_vector_b(1,1))**2)-array_q[3]**2
        eq_8 = torch.sum((r_02-self.gene_vector_b(0,2))**2)-array_q[4]**2
        eq_9 = torch.sum((r_12-self.gene_vector_b(1,2))**2)-array_q[5]**2
        #
        r_a_1,d_1,s_2_1 = self.gene_vector_r_a(r_00,r_10,i=0)
        r_a_2,d_2,s_2_2 = self.gene_vector_r_a(r_01,r_11,i=1)
        r_a_3,d_3,s_2_3 = self.gene_vector_r_a(r_02,r_12,i=2)        
        eq_10 = (r_10-r_00)@s_2_1
        eq_11 = (r_11-r_01)@s_2_2
        eq_12 = (r_12-r_02)@s_2_3
        
        eq_13 = (r_a_2-r_a_3)@d_1
        eq_14 = (r_a_1-r_a_3)@d_2
        eq_15 = (r_a_1-r_a_2)@d_3
        
        eq_16 = torch.sum((r_a_2-r_a_3)**2) - 3*self.a**2
        eq_17 = torch.sum((r_a_1-r_a_3)**2) - 3*self.a**2
        eq_18 = torch.sum((r_a_1-r_a_2)**2) - 3*self.a**2
        nonlinear_eqs = torch.stack([eq_13,eq_14,eq_15,eq_1,eq_4,eq_5,eq_10,eq_17,eq_18, eq_2,eq_11,eq_6,eq_7,eq_16, eq_3,eq_8,eq_9,eq_12])
        return nonlinear_eqs
    
    
    def solver(self,mask,r_all_vector,count_number=5):
        #First order Newton method, find the root, second order Newton method to solve the optimization problem, is essentially to find the root of the first derivative.
        r_all_vector = r_all_vector.to(device)  
        jacob = torch.autograd.functional.jacobian(self.nonlinear_eqs,r_all_vector)  
        for i in range(count_number):   
            g=self.nonlinear_eqs(r_all_vector)
            Q, R = torch.linalg.qr(jacob)
            diag_Q = torch.diagonal(Q)
            if torch.any(torch.abs(diag_Q) < 1e-5):
                return "Error"
            y_nonlinear_eqs = -self.nonlinear_eqs(r_all_vector)
            delta_r_all = torch.linalg.solve(R, Q.T @ y_nonlinear_eqs)
            r_all_vector += delta_r_all
            if torch.linalg.norm(delta_r_all,ord=2)<=1e-4:                
                break
            jacob +=(torch.outer(self.nonlinear_eqs(r_all_vector)-g - jacob@delta_r_all,delta_r_all)/delta_r_all@delta_r_all)[mask]
        # print(i)
        r_a_1,_,_ = self.gene_vector_r_a(r_all_vector[:3],r_all_vector[3:6],i=0)
        r_a_2,_,_ = self.gene_vector_r_a(r_all_vector[6:9],r_all_vector[9:12],i=1)
        r_a_3,_,_ = self.gene_vector_r_a(r_all_vector[12:15],r_all_vector[15:18],i=2)
        r_A =(r_a_1+r_a_2+r_a_3)/3 


        sys_A_y = r_a_2-r_a_3
        sys_A_y[:] = sys_A_y/torch.linalg.vector_norm(sys_A_y,ord=2)
        sys_A_x = (r_a_1-r_a_3)
        sys_A_z = torch.linalg.cross(sys_A_x,sys_A_y)
        sys_A_z[:] = sys_A_z/torch.linalg.vector_norm(sys_A_z,ord=2)
        sys_A_x[:] = torch.linalg.cross(sys_A_y,sys_A_z)
        
        R_A = torch.stack([sys_A_x,sys_A_y,sys_A_z]).T

        
        return r_A,R_A


def screw_matrix_z(alpha_z):
        screw_matrix_1 = torch.cos(alpha_z)
        screw_matrix_2 = torch.sin(alpha_z)
        screw_matrix = torch.tensor([[screw_matrix_1,-screw_matrix_2,0],
                                     [screw_matrix_2,screw_matrix_1,0],
                                     [0,0,1]],dtype = dtype,device = device)
        return screw_matrix
def screw_matrix_x(alpha_x):
        screw_matrix_1 = torch.cos(alpha_x)
        screw_matrix_2 = torch.sin(alpha_x)
        screw_matrix = torch.tensor([[1,0,0],
                                     [0,screw_matrix_1,-screw_matrix_2],
                                     [0,screw_matrix_2,screw_matrix_1]],dtype = dtype,device = device)
        return screw_matrix
    
def screw_matrix_y(alpha_y):
        screw_matrix_1 = torch.cos(alpha_y)
        screw_matrix_2 = torch.sin(alpha_y)
        screw_matrix = torch.tensor([[screw_matrix_1,0,screw_matrix_2],
                                     [0,1.,0],
                                     [-screw_matrix_2,0,screw_matrix_1]],dtype = dtype,device = device)
        return screw_matrix
    


