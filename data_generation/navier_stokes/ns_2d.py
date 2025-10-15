import torch
import torch.nn as nn
import argparse
import math
import os
import scipy.io
from timeit import default_timer
from random_fields import GaussianRF

import matplotlib.pyplot as plt
import matplotlib

# from drawnow import drawnow, figure

#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space (ensure input is real)
    w_h = torch.fft.rfft2(w0.real, norm="backward")

    #Forcing to Fourier space (ensure input is real)
    f_h = torch.fft.rfft2(f.real, norm="backward")

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction (for rfft2 output shape)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Wavenumbers for rfft2 grid (N x (N//2+1))
    k_x_rfft = k_x[:, :N//2 + 1]
    k_y_rfft = k_y[:, :N//2 + 1]

    #Negative Laplacian in Fourier space for rfft2 grid
    lap = 4*(math.pi**2)*(k_x_rfft**2 + k_y_rfft**2)
    lap[0,0] = 1.0 # Avoid division by zero for the zero frequency
    lap = lap.unsqueeze(0) # Add batch dimension for broadcasting

    #Dealiasing mask (needs adjustment for rfft2 shape)
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y_rfft) <= (2.0/3.0)*k_max, torch.abs(k_x_rfft) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.shape, record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap # Broadcasting should work now

        #Velocity field in x-direction = psi_y
        q_real = -2 * math.pi * k_y_rfft * psi_h.imag
        q_imag = 2 * math.pi * k_y_rfft * psi_h.real
        q_h = torch.complex(q_real, q_imag)
        q = torch.fft.irfft2(q_h, s=(N,N), norm="backward")

        #Velocity field in y-direction = -psi_x
        v_real = 2 * math.pi * k_x_rfft * psi_h.imag
        v_imag = -2 * math.pi * k_x_rfft * psi_h.real
        v_h = torch.complex(v_real, v_imag)
        v = torch.fft.irfft2(v_h, s=(N,N), norm="backward")

        #Partial x of vorticity
        w_x_real = -2 * math.pi * k_x_rfft * w_h.imag
        w_x_imag = 2 * math.pi * k_x_rfft * w_h.real
        w_x_h = torch.complex(w_x_real, w_x_imag)
        w_x = torch.fft.irfft2(w_x_h, s=(N,N), norm="backward")

        #Partial y of vorticity
        w_y_real = -2 * math.pi * k_y_rfft * w_h.imag
        w_y_imag = 2 * math.pi * k_y_rfft * w_h.real
        w_y_h = torch.complex(w_y_real, w_y_imag)
        w_y = torch.fft.irfft2(w_y_h, s=(N,N), norm="backward")

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y, norm="backward")

        #Dealias
        F_h = dealias * F_h # Apply dealiasing mask directly

        #Cranck-Nicholson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h) / (1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N), norm="backward")

            #Record solution and time
            sol[:, :, :, c] = w # Assign the batch to the corresponding slice
            sol_t[c] = t

            c += 1

    return sol, sol_t


def generate_ns_data(n_samples, s, record_steps, visc, T_final, bsize, f_func, device):
    """
    Helper function to generate a batch of Navier-Stokes solutions.
    """
    # Inputs
    a = torch.zeros(n_samples, s, s, device=device)
    # Solutions
    u = torch.zeros(n_samples, s, s, record_steps, device=device)
    
    c = 0
    t0 = default_timer()
    for j in range(n_samples // bsize):
        # Sample random fields
        w0 = GRF.sample(bsize)

        # Solve NS
        sol, sol_t = navier_stokes_2d(w0, f_func, visc, T_final, 1e-4, record_steps)
        
        # --- Immediate Check for NaNs/Infs ---
        if torch.isnan(sol).any() or torch.isinf(sol).any():
            print(f"!!! FATAL ERROR: NaNs or Infs generated in batch {j+1}. Halting generation. !!!")
            print("This batch contained invalid values and will be discarded.")
            # We must stop here to avoid corrupting the dataset.
            # Returning None indicates failure.
            return None, None, None

        a[c:(c+bsize),...] = w0
        u[c:(c+bsize),...] = sol

        c += bsize
        t1 = default_timer()
        print(f"Generated batch {j+1}/{n_samples // bsize}, total samples {c}, time elapsed {t1-t0:.2f}s")
    
    # The returned 'u' tensor should have the shape (n_samples, s, s, record_steps)
    # to be consistent with the training scripts.
    
    return a, u, sol_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Navier-Stokes data generation")
    parser.add_argument('--n_total', type=int, default=1000, help="Total number of samples desired.")
    parser.add_argument('--s', type=int, default=128, help="Resolution of the simulation grid (s x s).")
    parser.add_argument('--record_steps', type=int, default=20, help="Number of time steps to record.")
    parser.add_argument('--bsize', type=int, default=20, help="Batch size for generation.")
    parser.add_argument('--output_path', type=str, default='u.pt', help="Path to save the output data file.")
    parser.add_argument('--append', action='store_true', help="Append to existing data file if it exists.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, args.s, alpha=2.5, tau=7, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, args.s + 1, device=device)
    t = t[0:-1]
    X,Y = torch.meshgrid(t, t, indexing='ij')
    f_func = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    n_existing = 0
    existing_data = None
    if args.append and os.path.exists(args.output_path):
        print(f"Found existing data at {args.output_path}. Loading to append.")
        existing_data = torch.load(args.output_path, map_location=device)
        n_existing = existing_data.shape[0]
        print(f"Found {n_existing} existing samples.")
        if n_existing >= args.n_total:
            print("Number of existing samples already meets or exceeds target. Exiting.")
            exit()
    
    n_to_generate = args.n_total - n_existing
    
    if n_to_generate <= 0:
        print("No new samples to generate. Exiting.")
        exit()
        
    print(f"Need to generate {n_to_generate} new samples to reach a total of {args.n_total}.")
    
    if n_to_generate % args.bsize != 0:
        print(f"Warning: Number to generate ({n_to_generate}) is not divisible by batch size ({args.bsize}).")
        print(f"Adjusting to generate the next full batch: { (n_to_generate // args.bsize + 1) * args.bsize } samples.")
        n_to_generate = (n_to_generate // args.bsize + 1) * args.bsize
    
    # Generate new data
    _, new_u, _ = generate_ns_data(n_to_generate, args.s, args.record_steps, 1e-3, 20.0, args.bsize, f_func, device)

    # Check if generation failed
    if new_u is None:
        print("Data generation failed due to invalid values. Exiting without saving.")
        exit()

    # Combine with existing data if any
    if existing_data is not None:
        # The data from file is (B, H, W, T)
        # The new data new_u is also (B_new, H, W, T)
        final_u = torch.cat((existing_data, new_u), dim=0)
    else:
        final_u = new_u
        
    print(f"Total samples now: {final_u.shape[0]}")
    print(f"Saving combined data to {args.output_path}...")
    
    # Save the solution tensor 'u' directly
    torch.save(final_u, args.output_path)
    
    print("Data generation complete.")

# The old main execution block is removed or commented out.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ...
# scipy.io.savemat('ns_data.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
