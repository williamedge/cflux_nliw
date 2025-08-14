import numpy as np
import pymc3 as pm
import theano.tensor as tt


def allspeed_guess(rs_all, dtime, c_guess):
    v = np.array([rs_all[:,0,1]*dtime*c_guess,\
                 rs_all[:,0,2]*dtime*c_guess,\
                 rs_all[:,0,3]*dtime*c_guess,\
                 rs_all[:,1,2]*dtime*c_guess,\
                 rs_all[:,1,3]*dtime*c_guess,\
                 rs_all[:,2,3]*dtime*c_guess])
    return v


def theta_2_alldist(theta, xb, yb):
    theta_deg = np.rad2deg(theta)
    Phi2 = 90 - theta_deg

    # Distance from the beams to the origin in the wave direction
    d = xb*np.cos(np.deg2rad(Phi2)) + yb*np.sin(np.deg2rad(Phi2))  

    # Stack the distances
    distance = np.array([d[:,1] - d[:,0],\
                        d[:,2] - d[:,0],\
                        d[:,3] - d[:,0],\
                        d[:,2] - d[:,1],\
                        d[:,3] - d[:,1],\
                        d[:,3] - d[:,2]])
    return distance


def gen_pymc3_model(rs_all, dtime, obs_data, xb, yb):
    with pm.Model() as model:
        # Priors
        c = pm.Uniform('c', lower=0.0, upper=1.5)
        # c = pm.Lognormal('c', mu=0.35, sigma=0.75)
        theta = pm.VonMises('theta', mu=0.0, kappa=0.01)
        sigma = pm.HalfNormal('sigma', sd=5.0)

        # Likelihood
        mod_data = tt.as_tensor_variable(list(allspeed_guess(rs_all, dtime, c) - theta_2_alldist(theta, xb, yb)))
        ll = pm.Normal('y', mu=mod_data, sd=sigma, observed=obs_data)

        # Posterior
        trace = pm.sample(2000, tune=2000, chains=4, cores=4)
    return trace
