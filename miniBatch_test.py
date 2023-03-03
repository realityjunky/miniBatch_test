import numpy as np
import pymc as pm
import pickle


def mcmcVersion(dataPKL):
    """
    I have tested this to be ok.

    """

    with open(dataPKL, 'rb') as fh:
        dataDict = pickle.load(fh)

    with pm.Model() as m:

        m.add_coord('nC1', dataDict['covariate1'].unique(), mutable=True)
        m.add_coord('nC2', dataDict['covariate2'].unique(), mutable=True)
        m.add_coord('nC3', dataDict['covariate3'].unique(), mutable=True)
        m.add_coord('dimA', np.arange(dataDict['AD'].shape[1]), mutable=True)
        m.add_coord('dimB', np.arange(dataDict['AD'].shape[0]), mutable=True)

        TD_obs = pm.MutableData('TD_obs', dataDict['TD'], dims=('dimB', 'dimA'))
        AD_obs = pm.MutableData('AD_obs', dataDict['AD'], dims=('dimB', 'dimA'))
        c1_idx = pm.MutableData('c1_idx', dataDict['covariate1'].values, dims='dimA')
        c2_idx = pm.MutableData('c2_idx', dataDict['covariate2'].values, dims='dimA')
        c3_idx = pm.MutableData('c3_idx', dataDict['covariate3'].values, dims='dimA')

        mu_bc = pm.TruncatedNormal('mu_bc', mu=6, sigma=3, lower=3, shape=1)
        std_bc = pm.HalfNormal('std_bc', sigma=1, shape=1)

        mu_c1 = pm.TruncatedNormal('mu_c1', mu=mu_bc, sigma=std_bc, lower=3, dims='nC1')
        mu_c2 = pm.Normal('mu_c2', mu=0, sigma=1, dims='nC2')
        mu_c3 = pm.Normal('mu_c3', mu=0, sigma=1, dims='nC3')

        mu_p = pm.Deterministic('mu_p', mu_c1[c1_idx] + mu_c2[c2_idx] + mu_c3[c3_idx], dims='dimA')
        std_p = pm.HalfNormal('std_p', sigma=2, shape=1)
        ER_p = pm.Gamma('ER_p', alpha=mu_p ** 2 / std_p ** 2, beta=mu_p / std_p ** 2, dims='dimA')

        psi_p = pm.Beta('psi_p', alpha=2, beta=5, dims='dimA')
        AD_predicted = pm.ZeroInflatedBinomial('AD_predicted', psi=psi_p, n=TD_obs,
                                               p=pm.invlogit(-ER_p), observed=AD_obs)
        idata = pm.sample(300, init="adapt_diag", tune=300, cores=2, chains=2, target_accept=.9,
                          return_inferencedata=True)
