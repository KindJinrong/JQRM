import torch

dtype = torch.float64
device = 'cpu'

def trajectory(t, q_initial, q_target, T=1.0, device=device, dtype=dtype):
    """
    Calculate the trajectory parameters of a cubic polynomial: t (tensor): time vector (e.g., [0, T]), shape (N,) q_initial (tensor): initial position (6 dimensional), shape (6,) q_target (tensor): Target position (6-dimensional) with shape (6,) T (float): total time spent in motion device (str): device ('cpu' or 'cuda') dtype (torch.dtype): data type (e.g. torch.float32) Return: tensor: Trajectory of a 6-dimensional joint space, shape (N, 6)
    """
    q_initial = q_initial.to(device=device, dtype=dtype).unsqueeze(0)  
    q_target = q_target.to(device=device, dtype=dtype).unsqueeze(0)   
    t = t.to(device=device, dtype=dtype).unsqueeze(1)                

    a0 = q_initial 
    a3 = (2 * (q_initial - q_target)) / T**3 
    a2 = (-3 * (q_initial - q_target)) / T**2 
    a1 = torch.zeros_like(a0)  
    q_t = a3 * t**3 + a2 * t**2 + a1 * t + a0  

    return q_t


def v_trajectory(t, q_initial, q_target, T=1.0, device=device, dtype=dtype):
    """
    Calculate the trajectory parameters of a cubic polynomial: t (tensor): time vector (e.g., [0, T]), shape (N,) q_initial (tensor): initial position (6 dimensional), shape (6,) q_target (tensor): Target position (6-dimensional) with shape (6,) T (float): total time spent in motion device (str): device ('cpu' or 'cuda') dtype (torch.dtype): data type (e.g. torch.float32) Return: tensor: Trajectory of a 6-dimensional joint space, shape (N, 6)
    """
    q_initial = q_initial.to(device=device, dtype=dtype).unsqueeze(0)  
    q_target = q_target.to(device=device, dtype=dtype).unsqueeze(0)   
    t = t.to(device=device, dtype=dtype).unsqueeze(1)                

    a0 = q_initial 
    a3 = (2 * (q_initial - q_target)) / T**3 
    a2 = (-3 * (q_initial - q_target)) / T**2 
    a1 = torch.zeros_like(a0)  
    v_t = 3*a3 * t**2 + 2*a2 * t + a1   

    return v_t


def a_trajectory(t, q_initial, q_target, T=1.0, device=device, dtype=dtype):
    """
    Calculate the trajectory parameters of a cubic polynomial: t (tensor): time vector (e.g., [0, T]), shape (N,) q_initial (tensor): initial position (6 dimensional), shape (6,) q_target (tensor): Target position (6-dimensional) with shape (6,) T (float): total time spent in motion device (str): device ('cpu' or 'cuda') dtype (torch.dtype): data type (e.g. torch.float32) Return: tensor: Trajectory of a 6-dimensional joint space, shape (N, 6)
    """
    q_initial = q_initial.to(device=device, dtype=dtype).unsqueeze(0)  
    q_target = q_target.to(device=device, dtype=dtype).unsqueeze(0)   
    t = t.to(device=device, dtype=dtype).unsqueeze(1)                

    a0 = q_initial 
    a3 = (2 * (q_initial - q_target)) / T**3 
    a2 = (-3 * (q_initial - q_target)) / T**2 
    a1 = torch.zeros_like(a0)  
    a_t = 6*a3 * t + 2*a2    

    return a_t