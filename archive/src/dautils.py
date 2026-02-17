from shapely import Polygon, make_valid, GeometryCollection, MultiPolygon, wkt

import numpy as np
import pandas as pd
from pathlib import Path 
import datetime
from datetime import timedelta

from futils import forward_pass_farsite, cleanup_farsite_outputs
from tqdm import tqdm
from putils import calculate_max_area_geom, validate_geom
import traceback


BASE_DIR = "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored"

########## ENKF functions #########

def multipolygon_to_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

def state_to_geom(state):
    return validate_geom(Polygon(zip(state[::2], state[1::2])))

def geom_to_state(geom):
    geom = multipolygon_to_polygon(geom)
    coords = np.array(geom.exterior.coords[:-1])
    return coords.reshape(2 * len(coords), 1)


# def geom_to_state(geom):
#     return np.array(geom.exterior.coords[:-1]).reshape(2*len(geom.exterior.coords[:-1]), 1)

def load_enkf_inputs(path="enkf_inputs.npz"):
    data = np.load(path, allow_pickle=True)
    rng = np.random.default_rng()
    if "rng_state" in data:
        rng.bit_generator.state = data["rng_state"].item()
    return data, rng

################# ALIGN GEOMS ####################
def make_ccw(geom):
    if not geom.exterior.is_ccw:
        return geom.reverse()
    
    return geom

def interpolate_perimeter(vertices, dnumber):
    # Changes the number of vertices of the given set of vertices
    # if len(vertices) == dnumber:
    #     return vertices
    
    vertices = np.array(vertices)
    step_len = np.sqrt(np.sum(np.diff(vertices, 1, 0)**2, 1)) # length of each side
    step_len = np.append([0], step_len)
    cumulative_len = np.cumsum(step_len)
    interpolation_loc = np.linspace(0, cumulative_len[-1], dnumber)
    X = np.interp(interpolation_loc, cumulative_len, vertices[:,0])
    Y = np.interp(interpolation_loc, cumulative_len, vertices[:,1])

    return list(zip(X,Y))

def align_vertices(interpolated_vertices):
    minroll_lst = []
    
    aligned_vertices = [interpolated_vertices[0]]
    for i in range(len(interpolated_vertices)-1):
        right_vertices = interpolated_vertices[i+1]

        # Cycle right_vertices
        l2perroll = []
        for roll in range(len(interpolated_vertices[i])-1):
            diff = aligned_vertices[0] - right_vertices
            diff2sum = (diff[:,0]**2 + diff[:,1]**2).sum()

            # Calculate diff^2 in
            l2perroll.append(diff2sum)

            right_vertices = np.roll(right_vertices,1, axis=0)

        minroll_lst.append(np.argmin(l2perroll))

    for i in range(len(interpolated_vertices)-1):
        aligned_vertices.append(np.roll(interpolated_vertices[i+1], minroll_lst[i], axis=0))
    
    return aligned_vertices


def interpolate_geom(geom, vertex_count):
    interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count))
    if len(interpolated_geom.exterior.coords[:-1]) == vertex_count-1:
        interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count+1))
    if len(interpolated_geom.exterior.coords[:-1]) == vertex_count+1:
        interpolated_geom = Polygon(interpolate_perimeter((geom.exterior.coords), vertex_count-1))

    return interpolated_geom

def interpolate_geoms(geoms, vertex_count):
        
    interpolated_geoms = []
    for geom in geoms:
        interpolated_geoms.append(interpolate_geom(geom, vertex_count))
        
    return interpolated_geoms

def align_geoms(geoms, vertex_count): 
    '''
        Will align all the geometries based on geoms[0]
    '''
    
    # Calculate interpolated vertices first
    interpolated_geoms = interpolate_geoms(geoms, vertex_count)
    
    interpolated_vertices = [make_ccw(interpolated_geoms[0]).exterior.coords[:-1]]
    for geom in interpolated_geoms[1:]:
        interpolated_vertices.append(make_ccw(geom).exterior.coords[:-1])

    # for vertices in align_vertices(np.array(interpolated_vertices)):
    #     poly = Polygon(vertices)
    #     geom = interpolate_geom(validate_geom(poly), vertex_count)
            
    
    return [Polygon(vertices) for vertices in align_vertices(np.array(interpolated_vertices))]

#############################    
    
    
    
    
def align_states(state_lst, vertex_count=None):
    if vertex_count is None:
        vertex_count = max(len(st) for st in state_lst)//2
    x0 = state_lst[0][::2]
    y0 = state_lst[0][1::2]
    x1 = state_lst[1][::2]
    y1 = state_lst[1][1::2]

    geom0 = make_ccw(Polygon(zip(x0,y0)))
    geom1 = make_ccw(Polygon(zip(x1,y1)))

    geom0, geom1 = align_geoms([geom0, geom1], vertex_count)
    x,y = geom0.exterior.coords.xy
    x0 = x.tolist()[:-1]
    y0 = y.tolist()[:-1]
    state0 = xy_to_state(x0, y0)
    
    x,y = geom1.exterior.coords.xy
    x1 = x.tolist()[:-1]
    y1 = y.tolist()[:-1]
    state1 = xy_to_state(x1, y1)

    return [state0, state1]
    
def xy_to_state(x,y):
    ret = []
    for i in range(len(x)):
        ret.append(x[i])
        ret.append(y[i])

    return np.array(ret).reshape((2*len(x),1))


def resample_state_to_vertex_count(state, vertex_count):
    # state: (2*M, 1)
    geom = state_to_geom(state)
    geom = interpolate_geom(geom, vertex_count)  # returns polygon with ~vertex_count vertices
    return geom_to_state(geom)  # (2*vertex_count, 1)





def geom_bounds(g):
    minx, miny, maxx, maxy = g.bounds
    return (minx, miny, maxx, maxy)

def state_bounds(st):
    g = state_to_geom(st)
    return geom_bounds(g)



def adjusted_state_EnKF_farsite(initial_state, observation_state, observation_time,
                        X, n_states, n_output, n_vertex, n_samples, rng, 
                        sampled_wslst, sampled_wdlst, dt,
                        vsize, wsize, irwinid,
                               dist_res, perim_res):
    """
    initial_state: numpy array, output of geom_to_state()
    lcppath: filepath to existing landscape file or generate_landscape call
    n_samples: default 200 for 20 vertices (has to be >5x the number of vertices)
    sampled_wslst: random sample of wslst
    sampled_wdlst: random sample of wdlst
    X: covariance matrix (can be randomly or zero initialized)
    n_states: 40 (2x the number of vertices for (x,y) coordinates)
    """
    # Dimension fixes
    # --- 0) unwrap X if it accidentally came in as [array(...)] ---
    if isinstance(X, list):
        X = X[0]
    X = np.asarray(X)

    # --- 1) force fixed state dimension from n_vertex ---
    target_n_states = 2 * n_vertex

    initial_state = np.asarray(initial_state).reshape(-1, 1)
    observation_state = np.asarray(observation_state).reshape(-1, 1)

    # resample both to 2*n_vertex
    initial_state = resample_state_to_vertex_count(initial_state, n_vertex)
    observation_state = resample_state_to_vertex_count(observation_state, n_vertex)

    # now lock dimensions
    n_states = target_n_states
    n_output = target_n_states

    # ensure X matches
    if X.shape != (n_states, n_states):
        X = 1000.0 * np.eye(n_states)
    
    
    ### Ensemble calculation
    xkhat_ensemble = np.zeros((n_states, n_samples))
    zkphat_ensemble = np.zeros((n_states, n_samples))
    xkphat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble = np.zeros((n_output, n_samples))
    
    Xs = np.linalg.cholesky(X)  # square root of the covariance matrix
    # For each sample
    zero_samples = []
    
    # Generate lcp for initial_state
    initial_geom = state_to_geom(initial_state)

    if not Path("landscape.lcp").is_file():
        lcppath = generate_landscape(initial_geom, irwinid)
    else: lcppath = "landscape.lcp"


    # DEBUG
    # Before the loop
    init_geom = state_to_geom(initial_state)
    print("init bounds:", init_geom.bounds)
    
    # Observation sanity
    obs_geom = state_to_geom(observation_state)
    print("obs bounds:", obs_geom.bounds)


    ######################################## samplning loop
    
    for s in tqdm(range(n_samples)):

        # Retrieve sample based on covariance matrix to estimate the ensemble
        xkhat_ensemble[:,s:(s+1)] = initial_state + np.matmul(Xs, rng.normal(size=(n_states,1))) 

        ws = sampled_wslst[s]
        wd = sampled_wdlst[s]

        # Calculate the ensemble for the observations
        # ykhat is the predicted observation
        # ykhat_ensemble[:,s:(s+1)] = xy_to_state(*sample_xy(observation_state[::2], observation_state[1::2], rng, scale=vsize))
        
        ykhat_ensemble[:,s:(s+1)] = xkhat_ensemble[:,s:(s+1)] + rng.normal(0, scale=vsize, size=(n_output,1))
########################################

        # FARSITE calculation on sample
        total_dt = dt
        farsite_dt = pd.Timedelta(minutes=30)
        n_steps = int(total_dt / total_dt)
        
        farsite_params = {'windspeed': int(ws), 'winddirection': int(wd), 'dt': farsite_dt}
        input_poly = state_to_geom(xkhat_ensemble[:, s:(s+1)])
        start_time = observation_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print([farsite_params, start_time, lcppath, irwinid, dist_res, perim_res])
        for i in range(n_steps):
            forward_geom = forward_pass_farsite(
                poly=input_poly,
                params=farsite_params,
                start_time=start_time,
                irwinid=irwinid,
                lcppath=lcppath,
                dist_res=dist_res,
                perim_res=perim_res
            )
            input_poly = forward_geom
        

        
        cleanup_farsite_outputs(irwinid, BASE_DIR)
        dt = total_dt
        
        
        if forward_geom is None:
            zero_samples.append(s)
            # optional: print why/when it failed
            # print(f"Sample {s}: FARSITE returned None (ws={ws}, wd={wd})")
            continue
        
        print("forward bounds:", forward_geom.bounds)
        forward_state = geom_to_state(forward_geom) # Convert FARSITE output geometry to state variable

        # Compare vertex alignment between timestep t and t-1
        # Alignment to adjust forward state orientation or starting point of geometry
        aligned_states = align_states([initial_state, forward_state], vertex_count = n_vertex)

        zkphat_ensemble[:,s:(s+1)] = aligned_states[1]  # z is only ensembles
        
########################################  Data assimilation from here on out
    # valid = n_samples - len(zero_samples)
    # print("valid samples:", valid, "zero_samples:", len(zero_samples))
    # if valid <= 1:
    #     raise RuntimeError(f"Too few valid FARSITE samples ({valid}/{n_samples}). EnKF cannot proceed.")


    # # Calculate the mean of the non-zero ensembles
    # zkphat_mean = zkphat_ensemble.sum(axis=1, keepdims=True)/(n_samples - len(zero_samples))
    
    # # Fill in the zero samples with the mean (in the case that FARSITE doesn't calculate anything)
    # for s in zero_samples:
    #     zkphat_ensemble[:,s:(s+1)] = zkphat_mean
    
    # filled_counts = len(zero_samples)  # Filling zeroes

    # # Data assimilation
    # # zkphat_mean = zkphat_ensemble.mean(axis=1, keepdims=True)
    # ykhat_mean = ykhat_ensemble.mean(axis=1, keepdims=True)
    
    # # Calculate errors - step 1 of data assimilation
    # # zkphat_ensemble -= zkphat_mean
    # # ykhat_ensemble -= ykhat_mean
    # ezkphat_ensemble = np.zeros_like(zkphat_ensemble)

    # # Calculate the covariance matrices
    # for n in range(n_states):
    #     ezkphat_ensemble[n:(n+1),:] = zkphat_ensemble[n:(n+1),:] - zkphat_mean[n] + rng.normal(0, scale=wsize)  # + omega_k^j
    
    # eykhat_ensemble = np.zeros_like(ykhat_ensemble)
    # for n in range(n_output):
    #     eykhat_ensemble[n:(n+1),:] = ykhat_ensemble[n:(n+1),:] - ykhat_mean[n]

    # # Two types of the covariance matrices (Pzy and Py)
    # Pzy = 1/n_samples*np.matmul(ezkphat_ensemble, eykhat_ensemble.T)

    
    # Py = 1/n_samples*np.matmul(eykhat_ensemble, eykhat_ensemble.T)
    # Pyinv = np.linalg.pinv(Py)

    # # If n_states is too large for n_samples
    # # Works if n_states is >4x n_sampels, doesn't work otherwise
    # inv_product = np.matmul(Py, Pyinv)
    # if not np.allclose(inv_product, np.eye(n_output)): 
    #     print('Inverse calculation is incorrect')
    #     display(inv_product)
        
    # # warnings.warn('Not checking the inverse calculation')
    
    # # Compute estimated Kalman gain based on correlations
    # L_EnKF = np.matmul(Pzy, Pyinv)  # includes all covariance information
    
    # # compute mean valued state adjustment using measurement y(k)
    # # yk = geom_to_state(observation['geometry'], n_states, nvertex)
    # # Align actual measurements with the initial reference 
    # aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    # yk = aligned_states[1]  # Actual observation

    # # DEBUGGING
    # innovation = yk - ykhat_mean
    # delta = L_EnKF @ innovation

    # print("Calculating adjusted state")
    # adjusted_state = zkphat_mean + np.matmul(L_EnKF, yk - ykhat_mean)
    
    # # Compute the state adjustment ensembles to update state covariance matrix X
    # for j in range(n_samples):
    #     xkphat_ensemble[:,j:(j+1)] = zkphat_ensemble[:,j:(j+1)] + np.matmul(L_EnKF, yk - ykhat_ensemble[:,j:(j+1)])
    
    # # xkphat_ensemble -= xkphat_ensemble.mean(axis=1, keepdims=True)
    # xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    # exkphat_ensemble = np.zeros_like(xkphat_ensemble)
    # for n in range(n_states):
    #     exkphat_ensemble[n:(n+1), :] = xkphat_ensemble[n:(n+1),:] - xkphat_mean[n]
    
    # X = 1/n_samples*np.matmul(exkphat_ensemble, exkphat_ensemble.T) + 1e-10*np.eye(n_states)

    # # with initial and observed ensembles
    # # Validate adjusted state and reinterpolate
    # adjusted_geom = validate_geom(state_to_geom(adjusted_state))
    # adjusted_state = geom_to_state(adjusted_geom)
    # aligned_states = align_states([initial_state, adjusted_state], vertex_count=n_vertex)
    # adjusted_state = aligned_states[1]

    # # adjusted state is the initial state at t+1, X is the persisting covariance matrix
    # print("REACHED RETURN")
    # # return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble
    # return "SENTINEL"
    ########################################  Data assimilation from here on out
    
    # Boolean mask of valid ensemble members
    valid_mask = np.ones(n_samples, dtype=bool)
    valid_mask[zero_samples] = False
    Ne = int(valid_mask.sum())
    
    print("valid samples:", Ne, "zero_samples:", len(zero_samples))
    if Ne <= 1:
        raise RuntimeError(f"Too few valid FARSITE samples ({Ne}/{n_samples}). EnKF cannot proceed.")
    
    # Slice valid ensembles
    Z = zkphat_ensemble[:, valid_mask]   # (n_states, Ne)
    Y = ykhat_ensemble[:, valid_mask]    # (n_output, Ne)
    
    # Means
    zkphat_mean = Z.mean(axis=1, keepdims=True)     # (n_states, 1)
    ykhat_mean  = Y.mean(axis=1, keepdims=True)     # (n_output, 1)
    
    # Anomalies (additive model noise on the state anomalies, similar spirit to your loop)
    Z_anom = Z - zkphat_mean                         # (n_states, Ne)
    if float(wsize) > 0:
        Z_anom = Z_anom + rng.normal(0.0, float(wsize), size=Z_anom.shape)
    
    Y_anom = Y - ykhat_mean                          # (n_output, Ne)
    
    # Actual observation aligned to initial reference
    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk = aligned_states[1]                           # (n_output, 1)
    
    innovation = yk - ykhat_mean                     # (n_output, 1)
    
    # ---- Ensemble-space solve (Ne×Ne), assumes R = (vsize^2) I ----
    r2 = float(vsize) ** 2
    if r2 <= 0:
        raise ValueError("vsize must be > 0 (observation noise std).")
    rinv = 1.0 / r2
    
    # S = (Y^T R^{-1} Y) + (Ne-1) I   -> (Ne, Ne)
    S = rinv * (Y_anom.T @ Y_anom) + (Ne - 1) * np.eye(Ne)
    
    # Small ridge for numerical stability (cheap + helps rank issues)
    eps = 1e-10 * (np.trace(S) / S.shape[0])
    S_reg = S + eps * np.eye(Ne)
    
    # w = S^{-1} (Y^T R^{-1} innovation)
    rhs = rinv * (Y_anom.T @ innovation)             # (Ne, 1)
    w = np.linalg.solve(S_reg, rhs)                  # (Ne, 1)
    
    print("Calculating adjusted state (ensemble-space)")
    adjusted_state = zkphat_mean + (Z_anom @ w)      # (n_states, 1)
    
    # ---- Update analysis ensemble (optional but keeps your outputs consistent) ----
    # Deterministic member-wise update:
    # x_a^j = z^j + Z_anom * S^{-1} * (Y_anom^T R^{-1} (y - y^j))
    # where y^j are the predicted obs members (Y columns)
    xkphat_ensemble = np.zeros((n_states, n_samples))
    
    # Compute weights for all members at once:
    # W_all = S^{-1} (Y^T R^{-1} (y - Y))   -> (Ne, Ne)
    # innovation_all = y - Y (broadcast) -> (n_output, Ne)
    innovation_all = (yk - Y)                                # (n_output, Ne)
    rhs_all = rinv * (Y_anom.T @ innovation_all)            # (Ne, Ne)
    W_all = np.linalg.solve(S_reg, rhs_all)                 # (Ne, Ne)
    
    # Apply update
    xkphat_valid = Z + (Z_anom @ W_all)                     # (n_states, Ne)
    xkphat_ensemble[:, valid_mask] = xkphat_valid
    
    # For invalid members, fill with the mean analysis state (or keep forecast mean)
    xkphat_ensemble[:, ~valid_mask] = adjusted_state
    
    # ---- Update covariance X from analysis ensemble ----
    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    ex = xkphat_ensemble - xkphat_mean
    X = (ex @ ex.T) / (n_samples - 1) + 1e-10 * np.eye(n_states)
    
    # ---- Geometry validation / alignment (can be slow if self-intersections are nasty) ----
    adjusted_geom = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom)
    adjusted_state = align_states([initial_state, adjusted_state], vertex_count=n_vertex)[1]
    
    print("REACHED RETURN")
    return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble


###################
# SAMPLING ########
###################
def sample_geometry(geom, rng, sigma=1):
    sampled_vertices = []
    
    # Choose a random direction
    theta = rng.uniform(0,2*np.pi)

    for (x,y) in geom.exterior.coords[:-1]:
        mu=0
        
        randx = rng.normal(mu, sigma)
        randy = rng.normal(mu, sigma)
        
#         # Choose a normal random radius based on the given sigma
#         radius = abs(random.gauss(mu, sigma))
        
#         # Calculate x and y distance for the random
#         randx = radius*np.cos(theta)
#         randy = radius*np.sin(theta)
        
        sampled_vertices.append((x+randx, y+randy))

    sampled_vertices = np.array(sampled_vertices)
    return Polygon(sampled_vertices)

def sample_windspeed(loc, sigma, rng):
    ws = rng.normal(loc, sigma)
    if ws < 0:
        ws = 0
    return ws
def sample_winddirection(loc, sigma, rng):
    return np.fmod(rng.normal(loc, sigma)+360, 360)
