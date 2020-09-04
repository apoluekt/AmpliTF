# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import os
sys.path.append('../')
#os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Do not use GPU

import tensorflow as tf
import numpy as np
import argparse
import math

import amplitf.interface as atfi
atfi.backend_auto()
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.rootio as atfr
import amplitf.optimisation as atfo
import amplitf.likelihood as atfl
from iminuit import Minuit
from timeit import default_timer as timer

import amplitf.toymc as atft
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
from itertools import product

def couplingsign(spin, parity):
    return parity*(-1)**(1-spin/2)

def Couplings(res, spin, parity, pars):
    c = {} # (ldsst, ldst, lg)
    sign = couplingsign(spin, parity)
    ref = True if res == 'D2420' else False
    if ref:
        c[(-2, -2, -2)] = atfi.cast_complex(atfi.const(1.))
        c[(-2, -2, 2)] = -1.*c[(-2, -2, -2)]
        c[(-2, 2, -2)] = sign*c[(-2, -2, -2)]
        c[(-2, 2, 2)] = -1.*sign*c[(-2, -2, -2)]
    for l1, l2 in product([-2, 0, 2], [-2, 0]):
        if ref and l1 == -2 and l2 == -2: continue
        if spin == 0:
            if l2 ==  0:
                c[(l1, l2, -2)] = atfi.complex(pars[f'{res}_{l1}{l2}_re'], pars[f'{res}_{l1}{l2}_im'])
                c[(l1, l2, 2)] = -1.*c[(l1, l2, -2)]
            else:
                c[(l1, l2, -2)] = atfi.cast_complex(atfi.const(0.))
                c[(l1, l2, 2)] = c[(l1, l2, -2)]
                c[(l1, -1*l2, -2)] = c[(l1, l2, -2)]
                c[(l1, -1*l2, 2)] = c[(l1, l2, -2)]
        else:
            if l2 ==  0 and sign < 0:
                c[(l1, l2, -2)] = atfi.cast_complex(atfi.const(0.))
                c[(l1, l2, 2)] = c[(l1, l2, -2)]
            else:
                c[(l1, l2, -2)] = atfi.complex(pars[f'{res}_{l1}{l2}_re'], pars[f'{res}_{l1}{l2}_im'])
                c[(l1, l2, 2)] = -1.*c[(l1, l2, -2)]
                c[(l1, -1*l2, -2)] = sign*c[(l1, l2, -2)]
                c[(l1, -1*l2, 2)] = -1.*sign*c[(l1, l2, -2)]
    return c


def partial_integral(pdf, size):
    pdf2 = tf.reshape(pdf, [-1, size])
    return tf.reduce_mean(pdf2, axis=1)

if __name__ == '__main__' : 
    # Masses of the initial and final state particles
    mb   = 5.27932
    mdst = 2.010
    mdsst = 2.112
    mds = 1.968
    mdz  = 1.86483
    mk   = 0.493677
    mpi  = 0.13957061

    parser = argparse.ArgumentParser(description = 'B2DstDsstpi toy MC')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('-r', '--res', type=str, nargs='+', default=[])
    args = parser.parse_args() 

    strategy = tf.distribute.experimental.CentralStorageStrategy()

    res_names = ['D2420', 'D2430', 'D2460']
    spin_parity = [(2, 1), (2, 1), (4, 1)]
    masses = [atfi.const(2.4248), atfi.const(2.411), atfi.const(2.4607)]
    widths = [atfi.const(0.0336), atfi.const(0.309), atfi.const(0.0475)]
    ll = [2, 0, 2]

    if 'D2550' in args.res:
        res_names += ['D2550']; spin_parity += [(0, -1)]
        masses += [atfi.const(2.518)]; widths += [atfi.const(0.199)];
        ll += [1]
    if 'D2600' in args.res:
        res_names += ['D2600']; spin_parity += [(2, -1)]
        masses += [atfi.const(2.6419)]; widths += [atfi.const(0.149)];
        ll += [1]
    if 'D2740' in args.res:
        res_names += ['D2740']; spin_parity += [(4, -1)]
        masses += [atfi.const(2.751)]; widths += [atfi.const(0.102)];
        ll += [1]
    #res_names = ['D2430']
    #spin_parity = [(2, 1)]
    #masses = [atfi.const(2.411)]
    #widths = [atfi.const(0.309)]
    d = atfi.const(4.5)
    #mu = mdst*mdz/(mdst + mdz)
    I = atfi.complex(atfi.const(0.), atfi.const(1.))
    
    dltzphsp = DalitzPhaseSpace(mdst, mpi, mdsst, mb)
    angularphsp = RectangularPhaseSpace(((-1., 1.), (-math.pi, math.pi), (-1., 1.), (-math.pi, math.pi)))
    phsp = CombinedPhaseSpace(dltzphsp, angularphsp)

    def model(x, params, switches=len(res_names)*[1]):
        dlz = phsp.data1(x)
        ang = phsp.data2(x)

        m2dstpi = dltzphsp.m2ab(dlz)
        cos_th_dst =  dltzphsp.cos_helicity_ab(dlz)
        cos_th_d = angularphsp.coordinate(ang, 0)
        phi_d = angularphsp.coordinate(ang, 1)
        cos_th_ds  = angularphsp.coordinate(ang, 2)
        phi_ds = angularphsp.coordinate(ang, 3)
        
        #couplings = {}
        #for res, jp in zip(res_names, spin_parity):
        #    couplings[res] = Couplings(res, jp[0], jp[1], params)

        resonances = []
        for r, m, w, jp, l in zip(res_names, masses, widths, spin_parity, ll):
            couplings = Couplings(r, jp[0], jp[1], params)
            bw = atfd.breit_wigner_lineshape(m2dstpi, m, w, mdst, mpi, mdsst, mb, d, d, l, 1)
            resonances += [(bw, couplings, jp[0])]
        
        density = 0.
        for lambda_gamma in [-2, 2]:
            ampl = atfi.complex(atfi.const(0.), atfi.const(0.))
            idx = 0
            for res in resonances:
                amplres = atfi.complex(atfi.const(0.), atfi.const(0.))
                bw = res[0]
                couplings = res[1]
                j_dstst = res[2]
                for lambda_dst in [-2, 0, 2]:
                    for lambda_ in [-2, 0, 2]:
                        if switches[idx]:
                            h = atfi.cast_complex(atfk.wigner_small_d(
                                    atfi.acos(cos_th_dst), j_dstst, lambda_, lambda_dst))*\
                                atfi.exp(I*atfi.cast_complex(lambda_dst/2*phi_d))*\
                                atfi.cast_complex(atfk.wigner_small_d(
                                    atfi.acos(cos_th_d), 2, lambda_dst, 0))*\
                                atfi.exp(I*atfi.cast_complex(lambda_/2*phi_ds))*\
                                atfi.cast_complex(atfk.wigner_small_d(
                                    atfi.acos(cos_th_ds), 2, lambda_, -lambda_gamma))
                            amplres += h*couplings[(lambda_, lambda_dst, lambda_gamma)]
                ampl += amplres*bw
                idx += 1
            density += atfd.density(ampl)

        return density

    def momenta(x):
        dlz = phsp.data1(x)
        ang = phsp.data2(x)

        m2dstpi = dltzphsp.m2ab(dlz)
        cos_th_dst =  dltzphsp.cos_helicity_ab(dlz)
        cos_th_d = angularphsp.coordinate(ang, 0)
        phi_d = angularphsp.coordinate(ang, 1)
        cos_th_ds  = angularphsp.coordinate(ang, 2)
        phi_ds = angularphsp.coordinate(ang, 3)

        th_dst = atfi.acos(cos_th_dst)
        th_d = atfi.acos(cos_th_d)
        th_ds = atfi.acos(cos_th_ds)
        mdstpi = atfi.sqrt(m2dstpi)

        zeros = atfi.zeros(m2dstpi)
        ones = atfi.ones(m2dstpi)

        p4_dz, p4_pis = atfk.four_momenta_from_helicity_angles(
                                atfi.const(mdst)*ones, atfi.const(mdz)*ones, 
                                atfi.const(mpi)*ones, th_d, phi_d)
        p4_dst, p4_pi = atfk.four_momenta_from_helicity_angles(
                                atfi.const(mdstpi)*ones, atfi.const(mdst)*ones, 
                                atfi.const(mpi)*ones, th_dst, zeros)
        p4_ds, p4_gamma = atfk.four_momenta_from_helicity_angles(
                                atfi.const(mdsst)*ones, atfi.const(mds)*ones, 
                                zeros, th_ds, phi_ds)

        p_dsst = atfk.two_body_momentum(atfi.const(mb), atfi.const(mdsst), mdstpi)     
        p4_dsst = atfk.lorentz_vector(atfk.vector(zeros, zeros, -p_dsst), 
                                      atfi.sqrt(p_dsst**2 + mdsst**2))
        p4_dstpi = atfk.lorentz_vector(atfk.vector(zeros, zeros, p_dsst),
                                       atfi.sqrt(p_dsst**2 + mdstpi**2))

        p4_dz = atfk.rotate_lorentz_vector(p4_dz, zeros, -th_dst, zeros)
        p4_pis = atfk.rotate_lorentz_vector(p4_pis, zeros, -th_dst, zeros)
        p4_dz = atfk.boost_from_rest(p4_dz, p4_dst)
        p4_pis = atfk.boost_from_rest(p4_pis, p4_dst)
        
        p4_ds = atfk.boost_from_rest(p4_ds, p4_dsst)
        p4_gamma = atfk.boost_from_rest(p4_gamma, p4_dsst)
        p4_dst = atfk.boost_from_rest(p4_dst, p4_dstpi)
        p4_pi = atfk.boost_from_rest(p4_pi, p4_dstpi)
        p4_dz = atfk.boost_from_rest(p4_dz, p4_dstpi)
        p4_pis = atfk.boost_from_rest(p4_pis, p4_dstpi)

        return (p4_dstpi, p4_dsst, p4_dst, p4_pi, p4_dz, p4_pis, p4_ds, p4_gamma)

    def inv_masses(mom):
        p4_dstpi, p4_dsst, p4_dst, p4_pi, p4_dz, p4_pis, p4_ds, p4_gamma = mom
        m_dstpi = atfk.mass( p4_dstpi )
        m_dspi  = atfk.mass( p4_ds + p4_pi )
        m_dstds = atfk.mass( p4_dst + p4_ds )
        m_dsstpi = atfk.mass( p4_ds + p4_gamma + p4_pi )
        m_dstdspi = atfk.mass( p4_dst + p4_ds + p4_pi )
        #cosdst = atfk.cos_helicity_angle_dalitz(m_dstpi**2, m_dspi**2, mb, mdst, mpi, mdsst)
        #cosds = atfk.cos_helicity_angle_dalitz(mdsst**2, m_dstdspi**2, mb, 0.01, mds, m_dstpi)
        return (m_dstpi, m_dspi, m_dstds, m_dstdspi, m_dsstpi)
        
    def hel_angles(mom) :     
        p4_dstpi, p4_dsst, p4_dst, p4_pi, p4_dz, p4_pis, p4_ds, p4_gamma = mom
        th1_ds, phi1_ds, th1_dst, phi1_dst, th1_d, phi1_d = atfk.nested_helicity_angles([[[p4_dz, p4_pis], p4_pi], p4_dsst])
        return (th1_ds, phi1_ds, th1_dst, phi1_dst, th1_d, phi1_d)

    def kin(x):
        mom = momenta(x)
        l = []
        l += inv_masses(mom)
        l += hel_angles(mom)
        return l

    pars = [
        atfi.FitParameter('D2420_-20_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2420_-20_im', 0.5, -10., 10., 0.01),
        atfi.FitParameter('D2420_0-2_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2420_0-2_im', 0.5, -10., 10., 0.01),
        atfi.FitParameter('D2420_00_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2420_00_im', 0.5, -10., 10., 0.01),
        atfi.FitParameter('D2420_2-2_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2420_2-2_im', 0.5, -10., 10., 0.01),
        atfi.FitParameter('D2420_20_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2420_20_im', 0.5, -10., 10., 0.01),

        atfi.FitParameter('D2430_-2-2_re', 1.0, -10., 10., 0.01), atfi.FitParameter('D2430_-2-2_im', 1.75, -10., 10., 0.01),
        atfi.FitParameter('D2430_-20_re', 1.75, -10., 10., 0.01), atfi.FitParameter('D2430_-20_im', 1.75, -10., 10., 0.01),
        atfi.FitParameter('D2430_0-2_re', 1.75, -10., 10., 0.01), atfi.FitParameter('D2430_0-2_im', 1.75, -10., 10., 0.01),
        atfi.FitParameter('D2430_00_re', 0.75, -10., 10., 0.01), atfi.FitParameter('D2430_00_im', 0.75, -10., 10., 0.01),
        atfi.FitParameter('D2430_2-2_re', 0.75, -10., 10., 0.01), atfi.FitParameter('D2430_2-2_im', 0.75, -10., 10., 0.01),
        atfi.FitParameter('D2430_20_re', 0.75, -10., 10., 0.01), atfi.FitParameter('D2430_20_im', 0.75, -10., 10., 0.01),

        atfi.FitParameter('D2460_-2-2_re', 1.5, -10., 10., 0.01), atfi.FitParameter('D2460_-2-2_im', 1.5, -10., 10., 0.01),
        atfi.FitParameter('D2460_0-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2460_0-2_im', 0.5, -10., 10., 0.01),
        atfi.FitParameter('D2460_2-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2460_2-2_im', 0.5, -10., 10., 0.01),
    ]
    if 'D2550' in args.res:
        pars += [
            atfi.FitParameter('D2550_-20_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2550_-20_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2550_00_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2550_00_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2550_20_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2550_20_im', 0.5, -10., 10., 0.01),
        ]
    if 'D2600' in args.res:
        pars += [
            atfi.FitParameter('D2600_-2-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2600_-2-2_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2600_0-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2600_0-2_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2600_2-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2600_2-2_im', 0.5, -10., 10., 0.01),
        ]
    if 'D2740' in args.res:
        pars += [
            atfi.FitParameter('D2740_-2-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_-2-2_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2740_-20_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_-20_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2740_0-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_0-2_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2740_00_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_00_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2740_2-2_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_2-2_im', 0.5, -10., 10., 0.01),
            atfi.FitParameter('D2740_20_re', 0.5, -10., 10., 0.01), atfi.FitParameter('D2740_20_im', 0.5, -10., 10., 0.01),
        ]
    
    initpars = {p.name : atfi.const(p.init_value) for p in pars}
    """
    couplings = {}
    for res, jp in zip(res_names, spin_parity):
        couplings[res] = Couplings(res, jp[0], jp[1], initpars)

    for k, v in couplings.items():
        print(k)
        for l, h in v.items():
            print(l, h)
    
    atfi.set_seed(args.seed)
    """
    def gen_model(x, switches=len(res_names)*[1]):
        return model(x, initpars, switches=switches)

    #norm_sample = phsp.uniform_sample(500000)
    norm_sample = np.load('norm_sample.npy', allow_pickle=True)

    maximum = atft.maximum_estimator(gen_model, phsp, 100000) * 1.5
    print(maximum)
    toy_sample = 9000
    gen_sample = atft.run_toymc(gen_model, phsp, toy_sample, maximum, chunk = 1000000)
    gen_sample = gen_sample[:,:6]
    #helangle = np.stack([dltzphsp.cos_helicity_ab(phsp.data1(gen_sample))], axis=1)
    #kinematics = np.stack(kin(gen_sample), axis=1)
    #arr = np.concatenate([gen_sample, helangle, kinematics], axis=1)
    #branches = ['m2dp', 'm2ppi'] + [ f'w{i}' for i in range(7) ]
    #branches = ['m2dstpi', 'mdsstpi', 'costhd', 'phid', 'costhds', 'phids']
    #branches = ['m2dstpi', 'm2dspi', 'costhd', 'phid', 'costhds', 'phids', 'costhdst']
    #branches += ['m_dstpi', 'm_dspi', 'm_dstds', 'm_dstdspi', 'm_dsstpi']
    #branches += ['th_ds', 'phi_ds', 'th_dst', 'phi_dst', 'th_d', 'phi_d']
    #branches += ['th_r', 'phi_r', 'th_dst', 'phi_dst', 'th_d', 'phi_d', 'th_ds', 'phi_ds']
    #atfr.write_tuple('test_gen.root', arr, branches)
    
    ffi = atfo.calculate_fit_fractions(gen_model, norm_sample)
    print([np.asscalar(f.numpy()) for f in ffi])

    if args.fit:
        integ_points = 50
        int_sample = np.linspace(-math.pi, math.pi, integ_points, endpoint = False)
        data_sample_2 = np.repeat(gen_sample[:,0:-1], int_sample.shape[0], axis=0)   
        int_sample2  = np.reshape(np.tile(int_sample, gen_sample.shape[0]), (-1,1))   
        data_sample  = np.concatenate( [data_sample_2, int_sample2], axis=1 )
        
        global_batch_size = data_sample.shape[0]
        tf_dataset = tf.data.Dataset.from_tensor_slices(data_sample).batch(global_batch_size)
        dist_dataset = strategy.experimental_distribute_dataset(tf_dataset)
        dist_values = iter(dist_dataset).get_next()

        @atfi.function
        def nll(data, norm, params) : 
            return atfl.unbinned_nll(model(data, params), atfl.integral(model(norm, params)))

        float_pars = [p for p in pars if p.floating() ]
        kwargs = { p.name : p() for p in float_pars }
        #print("loss ", nll(gen_sample, norm_sample, kwargs))
        #loss = strategy.run(nll, args=(next(iter(dist_dataset)), norm_sample, kwargs,))
        #print('Per replica loss is ', loss)
        #val = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        #print('sum is', val, type(val))
        
        #### Run minuit code
        #### Here for test purposes
        ####
        #@atfi.function
        def func(par) :
            for i,p in enumerate(float_pars) : p.update(par[i])
            kwargs = { p.name : p() for p in float_pars }
            func.n += 1
            nll_per_replica = strategy.run(nll, args=(dist_values, norm_sample, kwargs))
            nll_val = strategy.reduce(tf.distribute.ReduceOp.SUM, nll_per_replica, axis=None) 
            if func.n % 100 == 0 : print(func.n, nll_val, par)
            return nll_val
        
        #@atfi.function
        def gradient(par) :
            for i, p in enumerate(float_pars): p.update(par[i])
            kwargs = { p.name : p() for p in float_pars }
            float_vars = [ i() for i in float_pars ]
            gradient.n += 1 
            def replica_gt(d):
                with tf.GradientTape() as gt : 
                    gt.watch( float_vars )
                    nll_val = nll(d, norm_sample, kwargs)
                g = gt.gradient(nll_val, float_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                #g_val = [ i.numpy() for i in g ]
                return g
            return [strategy.reduce('SUM', rep, axis=None).numpy() for rep in strategy.run(replica_gt, args=(dist_values,))]
        
        func.n = 0 
        gradient.n = 0 
        start = [ p.init_value for p in float_pars ]
        error = [ p.step_size for p in float_pars ]
        limit = [ (p.lower_limit, p.upper_limit) for p in float_pars ]
        name = [ p.name for p in float_pars ]
        use_gradient = True
        if use_gradient : 
            minuit = Minuit.from_array_func(func, start, error = error, limit = limit, name = name, grad = gradient, errordef = 0.5)
        else : 
            minuit = Minuit.from_array_func(func, start, error = error, limit = limit, name = name, errordef = 0.5)

        start = timer()
        minuit.migrad()
        end = timer()
        
        par_states = minuit.get_param_states()
        f_min = minuit.get_fmin()

        results = { "params" : {} } # Get fit results and update parameters
        for n, p in enumerate(float_pars) :
            p.update(par_states[n].value)
            p.fitted_value = par_states[n].value
            p.error = par_states[n].error
            results["params"][p.name] = (p.fitted_value, p.error)

        # return fit results
        results["loglh"] = f_min.fval
        results["iterations"] = f_min.ncalls
        results["func_calls"] = func.n
        results["grad_calls"] = gradient.n
        results["time"] = end-start
        print(results)
        exit()

        #######
        @atfi.function
        def nll(data, norm, params):
            return atfl.unbinned_nll(partial_integral(model(data, params), integ_points), atfl.integral(model(norm, params)))

        float_pars = [p for p in pars if p.floating() ]
        kwargs = { p.name : p() for p in float_pars }
        loss = nll(data_sample, norm_sample, kwargs)
        print('loss is ', loss, type(loss))

        result = atfo.run_minuit(nll, pars, args=(data_sample, norm_sample), use_gradient=True)
        print(result)
        
        fittedpars = { k : atfi.const(v[0]) for k,v in result["params"].items() }

        def fit_model(x, switches=len(res_names)*[1]):
            return model(x, fittedpars, switches=switches)

        ff = atfo.calculate_fit_fractions(fit_model, norm_sample)
        
        atfo.write_fit_results(pars, result, 'results.txt')
        exit()
        #######
