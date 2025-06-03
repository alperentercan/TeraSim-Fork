import cvxpy as cp  # inside terasim-cosim env
import numpy as np


class StatesLong:
    def __init__(self, headway, v_long, v_lead):
        self.hw = headway # headway between ego and lead vehicle
        self.v_long = v_long # longitudinal speed of ego
        self.v_lead = v_lead # longitudinal speed of lead vehicle


class AlphaFilter:
    def __init__(self, H_max, h_max, H_xu, h_xu):
        self.f = True
        self.H_max = H_max
        self.h_max = h_max
        self.H_xu = H_xu
        self.h_xu = h_xu
        self.u_pre = [0, 0]  # previous input

        self.pos_ind = H_max[:, -1] > 0
        self.zero_ind = H_max[:, -1] == 0
        self.neg_ind = H_max[:, -1] < 0
        if not np.any(self.pos_ind):
            self.pos_ind = None
        if not np.any(self.neg_ind):
            self.neg_ind = None

    def get_max_alpha(self, x):
        '''
        Compute the maximal alpha given the current state

        Inputs: 
        - x (np.ndrray): the state of one subsystem, 3x1
        
        Return: 
        - alpha : the maximal alpha at x
        '''
        # H, h are the constraints of the admissible input set
        A = self.H_max[:, -1]
        b = self.h_max - self.H_max[:, 0:-1] @ x

        alpha_max = np.inf
        alpha_min = -np.inf
        
        if self.pos_ind is not None:
            alpha_max = np.min(b[self.pos_ind].squeeze() / A[self.pos_ind])
        if self.neg_ind is not None:
            alpha_min = np.max(b[self.neg_ind].squeeze() / A[self.neg_ind])

        if (alpha_max < alpha_min) or (np.any(b[self.zero_ind] < -1e-6)):
            alpha_max = -np.inf
            alpha_min = -np.inf

        return alpha_max, alpha_min

    def qp(self, x:np.ndarray, u_ref:float) -> float:
        '''
        Solve the QP program in one subsystem.

        Inputs:
        - x (np.array): the states of one subsystem (i.e. headway, headway rate, alpha)
        - u_ref (float) :the reference input
        
        Return: 
        u_corr (float): the supervised input
        '''

        u = cp.Variable(2) # why was it 2?

        H = self.H_xu[:, 3:]
        h = self.h_xu - self.H_xu[:, 0:3] @ x
        prob = cp.Problem(cp.Minimize((u[0] - u_ref) ** 2),
                          [H @ u <= h.squeeze()])
        # prob.solve()

        prob.solve(cp.PROXQP)
        try:
            return u.value[0]
        except TypeError:
            print("QP solver error!", x)
            return u_ref


    def eval(self, x:np.ndarray, u_ref:float) -> float:
        '''
        Project the reference input to the admissible input set at x.

        Inputs: 
        - xy (list) : the states of the 4d system [headway, headway rate]
        - u_ref (list) : the reference input - acceleration
        
        Return: 
        - u (list) : the supervised input, 1x2
        '''
        alpha_x, _ = self.get_max_alpha(x)

        if alpha_x == -np.inf:
            # the reference input is admissable
            ux = u_ref
        else:
            # the reference input is not admissable, solve the QP
            ux = self.qp(np.vstack((x, alpha_x)), u_ref)

        self.u_pre = ux 
        return ux