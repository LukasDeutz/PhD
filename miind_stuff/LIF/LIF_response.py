'''
Created on 3 Jan 2019

@author: lukas
'''

import os
import numpy as np
import h5py as h5
import miind_api as api
import mesh3
import mpmath as mp 
import itertools as it
import matplotlib.pyplot as plt

from scipy.optimize import brentq, root
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter


from miind_library.generate_simulation import generate_model, generate_empty_fid, generate_matrix

import nest
from fileinput import filename

class LIFMeshGenerator:
    
    def __init__(self, basename, model_param, miind_param):
                                        
        self.basename = basename
        self.model_param    = model_param
        self.miind_param    = miind_param

        self.tau   = self.model_param['tau'] # membrane time constant in s
        self.v_th  = self.model_param['v_th'] # threshold in V
                    
        self.v_rev   = self.model_param['v_rev']
        self.v_reset = self.model_param['v_reset']
        self.v_min   = self.model_param['v_min']
        self.hE = self.model_param['hE']
                        
        self.N_grid  = self.miind_param['N_grid'] # number of points in the interval [V_res, self.v_th); e.g if self.v_min = self.v_th, the grid holds double this number of bins
        self.dt      = self.miind_param['dt'] # timestep for each bin
        self.strip_w = self.miind_param['w_strip']# arbitrary value for strip width
                
    def generateLIFMesh(self):
                                      
        if self.v_min > self.v_reset:
            raise ValueError ("v_min must be less than v_reset.")

        with open(self.basename + '.mesh','w') as meshfile:
            meshfile.write('ignore\n')
            meshfile.write('{}\n'.format(self.dt))
                                    
            ts = self.dt * np.arange(self.N_grid)
                                                                                    
            pos_vs = self.v_rev + (self.v_th - self.v_rev)*np.exp(-ts/self.tau)
            
            # Add additional cells
            # The first will become the threshold cell
            pos_vs = np.insert(pos_vs, 0, self.v_th + 2*self.hE)
            # And the second cell will serve as the reversal cell when mass reaches the stationary cell
            pos_vs = np.insert(pos_vs, 0, self.v_th + 4*self.hE)
            
            neg_vs = self.v_rev + (self.v_min - self.v_rev)*np.exp(-ts/self.tau)

            # right border stat cell
            self.v_stat_rb = pos_vs[-1]
            # left border stat cell
            self.v_stat_lb = neg_vs[-1]

            if len(neg_vs) > 0:
                for v in neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')

                for v in pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')
                meshfile.write('end')

        return self.basename + '.mesh'

    def generateLIFStationary(self):
        statname = self.basename + '.stat'
        
        v0 = self.v_stat_lb
        v1 = self.v_stat_rb

        with open(statname,'w') as statfile:
            statfile.write('<Stationary>\n')
            format = "%.9f"
            statfile.write('<Quadrilateral>')
            statfile.write('<vline>' +  str(v0) + ' ' + str(v0) + ' ' +  str(v1) + ' ' + str(v1) + '</vline>')
            statfile.write('<wline>' +  str(0)     + ' ' + str(self.strip_w)     + ' ' +  str(self.strip_w)      + ' ' + str(0)      + '</wline>')
            statfile.write('</Quadrilateral>\n')
            statfile.write('</Stationary>')

    def generateLIFReversal(self):
        revname = self.basename  + '.rev'
        m=mesh3.Mesh(self.basename + '.mesh')

        with open(revname,'w') as revfile:
            revfile.write('<Mapping type=\"Reversal\">\n')
            for i in range(1,len(m.cells)):
                revfile.write(str(i) + ',' + str(0))
                revfile.write('\t')
                revfile.write(str(0) + ',' + str(0))
                revfile.write('\t')
                revfile.write(str(1.0) + '\n')
            revfile.write('</Mapping>')

class LIF_Response():
        
    def __init__(self, model_param):
        
        self.model_param = model_param
        self.LIF_miind = None
        self.LIF_mc    = None
        self.LIF_nest  = None
                                        
    def run_miind(self, miind_param):
        ''' '''
        
        self.LIF_miind = LIF_Response_Miind(self.model_param, miind_param)
        self.LIF_miind.simulate()
        
        return
                                        
    def run_mc(self, mc_param):
        ''' '''
        
        self.LIF_mc = LIF_Response_MC(self.model_param, mc_param)
        self.LIF_mc.simulate()
        
        return

    def run_nest(self, nest_param):
        
        self.LIF_nest = LIF_Response_Nest(self.model_param, nest_param)
        self.LIF_nest.simulate()
        
        return
        
    def save(self, filename, mode = 'w'):
        
        with h5.File(filename, mode) as hf5:
            
            # save parameter
            hf5.create_group('model_parameter')
            hf5['model_parameter'].update(self.model_param)            
        
            if not self.LIF_miind is None:
                self.LIF_miind.save(hf5)
    
            if not self.LIF_mc is None:
                self.LIF_mc.save(hf5)
        
            if not self.LIF_nest is None:
                self.LIF_nest.save(hf5)
        
        return

class LIF_Response_Nest():
    '''Response of a population of non-interacting LIF neurons to Poisson input simulated with Nest'''        
    
    def __init__(self, 
                 model_param,
                 nest_param):
        '''Simulates response of non-interacting LIF neurons to external input using Miind
        :param dict model_param: neuron model parameter
        :param miind_param: parameter required to run miind simulation
        '''        
        #os.chdir(directory)
        
        self.model_param = model_param
        self.nest_param  = nest_param

    
    def nest_model_parameter(self):
        '''Rename model parameter for compatibility with nest'''
        
        param = {}
    
        param['V_reset'] = self.model_param['v_reset']
        param['V_th']    = self.model_param['v_th']
        param['tau_m']   = self.model_param['tau']*1.e3           
        param['V_min']   = self.model_param['v_min']            
        param['E_L']     = self.model_param['v_rev']
        param['V_m']     = 0.0
        param['C_m']     = 1.0

    #     param['tau_ref'] = 0.0
        return param    
                
    def simulate(self):
        '''Simulate respone of non-interacting population LIF neurons to Poisson input using Miind'''
    
        # nest parameter                     
        N         = self.nest_param['N_trial'] # number of trials
        dt        = self.nest_param['dt'] # time step
        t_end     = self.nest_param['t_end']
        dt_report = self.nest_param['dt_report']
        
        
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': dt*1.e3})
                    
        pop = nest.Create('iaf_psc_delta', N, params = self.nest_model_parameter())
    
        # Poisson input
        KE = self.model_param['KE']
        hE = self.model_param['hE']
        rE = self.model_param['r_ext_E']

        poisson_E = nest.Create("poisson_generator")
        nest.SetStatus(poisson_E, {'rate': KE*rE})                
        nest.Connect(poisson_E, pop, syn_spec = {'weight': hE})
                                
        if self.model_param['N_inp'] == 2:

            KI = self.model_param['KI']
            hI = self.model_param['hI']
            rI = self.model_param['r_ext_I']

            poisson_I = nest.Create("poisson_generator")
            nest.SetStatus(poisson_I, {'rate': KI*rI})                
            nest.Connect(poisson_I, pop, syn_spec = {'weight': hI})
          
        # measure voltage traces            
        mm = nest.Create('multimeter', params = {"withtime": False, 'record_from':['V_m'], 'interval' : dt_report*1.e3})        
        nest.Connect(mm, pop)
                        
        # detect spike times
        sd = nest.Create('spike_detector', params={"withgid": True, "withtime": True})                
        nest.Connect(pop, sd)
                        
        # run simulation
        nest.Simulate(t_end*1.e3)
        
        # get matrix with voltage traces for all trials
        data_mm = nest.GetStatus(mm, keys = 'events')[0]        
        self.v_mat = np.reshape(data_mm['V_m'], (int(t_end/dt_report - 1), N))
    
        # get array with spike times
        data_sd = nest.GetStatus(sd, keys = 'events')[0] 
        self.spike_times = data_sd['times']*1.e-3
        
        self.t_report = np.arange(0, t_end-dt_report, dt_report)
        
        return

    def save(self, h5f):
            
        density = self.get_density_dict()
        rate    = self.get_rate_dict()
                              
        # save density
        for key, value in density.items():
             
            path = 'nest/density/' + key
            h5f.create_dataset(path, data = value)
        
        # save rate
        for key, value in rate.items():
             
            path = 'nest/rate/' + key
            h5f.create_dataset(path, data = value)
            
        # save parameter                            
        h5f.create_group('nest/parameter')        
        h5f['nest/parameter'].update(self.nest_param)

        return

    def calculate_density(self):
        '''Caculate the density of the membrane potential from the voltage traces for different trials''' 
                
        n_vbins   = self.nest_param['n_vbins']
        v_min = self.model_param['v_min']
        v_th = self.model_param['v_th']
                            
        self.bins_v = np.linspace(v_min, v_th, n_vbins, endpoint = True)
        
        # density matrix                            
        self.rho_mat = np.zeros((np.size(self.v_mat, 0), n_vbins -1 ))
                
        # calculate density                
        for i, v in enumerate(self.v_mat):
            
            rho_t, _ = np.histogram(v, self.bins_v, density = True)            
            self.rho_mat[i, :] = rho_t
                
        return
        
    def calculate_rate(self):
        '''Caculate the density of the membrane potential from the voltage traces for different trials''' 
                        
        N_trial   = self.nest_param['N_trial']
        dt_report = self.nest_param['dt_report']
        t_end     = self.nest_param['t_end']

        bins = np.arange(0, t_end, dt_report)

        spike_counts, _ = np.histogram(self.spike_times, bins)
        self.r  = spike_counts/N_trial/dt_report
                                
    def get_density_dict(self):
        '''Returns dictionary with density, v_bins, and v'''
        
        self.calculate_density()
                
        density = {}
        density['t']   = self.t_report
        density['rho'] = self.rho_mat
        
        dv = (self.bins_v[1] - self.bins_v[0])/2        
        density['v']    = self.bins_v[:-1] + dv
        density['bins'] = self.bins_v
        
        return density
                                
    def get_rate_dict(self):
        
        self.calculate_rate()
        
        rate = {}
        rate['t'] = self.t_report
        rate['r'] = self.r        
                
        return rate 

        
class LIF_Response_Miind(): 
    '''Response of a population of non-interacting LIF neurons to Poisson input simulated with Miind'''        
    def __init__(self, 
                 model_param,
                 miind_param):
        '''Simulates response of non-interacting LIF neurons to external input using Miind
        :param dict model_param: neuron model parameter
        :param miind_param: parameter required to run miind simulation
        '''        
        #os.chdir(directory)
        
        self.basename    = miind_param['basename']
        self.model_param = model_param
        self.miind_param = miind_param
        
        self.node = 'LIF0'
        
    def simulate(self, 
                 overwrite = True, 
                 enable_mpi = False, 
                 enable_openmp = True, 
                 enable_root = False):
        '''Simulate respone of non-interacting population LIF neurons to Poisson input using Miind'''
                                        
        self.generate_model()

        self.write_xml()

        # initialize simulation
        self.sim = api.MiindSimulation('./' + self.xml_file, self.miind_param['submit'])
        # submit simulation
        self.sim.submit(overwrite, enable_mpi, enable_openmp, enable_root)
        # run simulation
        self.sim.run()


    def save(self, 
             h5f):
        '''Save simulation resuls in as hdf5
        :param str filename: filename
        :param dict attributes: dictionary with meta data'''         
        
        rate    = self.get_rate_dict()
        density = self.get_density_dict()
                        
        # save density
        for key, value in density.items():
             
            path = 'miind/density/' + key
            h5f.create_dataset(path, data = value)
        
        # save rate
        for key, value in rate.items():
             
            path = 'miind/rate/' + key
            h5f.create_dataset(path, data = value)
        
        # save parameter        
        h5f.create_group('miind/parameter')        
        h5f['miind/parameter'].update(self.miind_param)
                                                                                                 
        return 

    def generate_mesh(self):
        '''Generate mesh'''
        
        mesh_generator = LIFMeshGenerator(self.basename, 
                                          self.model_param, 
                                          self.miind_param)
         
        mesh_generator.generateLIFMesh()
        mesh_generator.generateLIFStationary()
        mesh_generator.generateLIFReversal()        

    def generate_model(self):        
        '''Generate model fid and matrix file
        :param dict param: param'''
                
        # generate mesh        
        self.generate_mesh()
        
        # generate model from mesh
        v_reset   = self.model_param['v_reset'] # reset potential
        v_th      = self.model_param['v_th'] # threshold potential
        hE        = self.model_param['hE'] # efficacy        
        h_arr     = self.model_param['h_arr'] # efficacy
        N_mc      = self.miind_param['N_mc'] # number of monte carlo samples
        uac       = self.miind_param['uac'] # caculate transition matrix geomtrically
                
        eps = 0.01*hE
        
        generate_model(self.basename, v_reset, v_th, eps = eps)
        generate_empty_fid(self.basename)    
                
        for i, h in enumerate(h_arr):
            matrix = generate_matrix(self.basename, h, N_mc, uac)
            self.miind_param['mat' + str(i)] = matrix
                
        return 

    def write_xml(self):
        '''Write xml file for given param'''
    
        self.xml_file = self.basename + '.xml' 
    
        if self.model_param['N_inp'] == 1:
            xml_skellet = """<Simulation>
<WeightType>DelayedConnection</WeightType>
<Algorithms>
<Algorithm type="MeshAlgorithm" name="LIF" modelfile="{basename}.model" >
<TimeStep>{dt}</TimeStep>
<MatrixFile>{mat0}</MatrixFile>
</Algorithm>
<Algorithm type="RateFunctor" name="ExtInp">
<expression>{r_ext_E}</expression>
</Algorithm>
</Algorithms>
<Nodes>
<Node algorithm="LIF" name="LIF0" type="EXCITATORY_DIRECT" />
<Node algorithm="ExtInp" name="ExcInp0" type="NEUTRAL" />
</Nodes>
<Reporting>
<Density node="LIF0" t_start="0.0" t_end="{t_end}" t_interval="{t_report}" />
<Rate node="LIF0" />
<!-- Display node="LIF0" -->
</Reporting>
<Connections>
<Connection In="ExcInp0" Out="LIF0">{K} {hE} 0</Connection>
</Connections>
<SimulationIO>
<SimulationName>{basename}</SimulationName>
<OnScreen>FALSE</OnScreen>
<WithState>TRUE</WithState>
<WriteNet>FALSE</WriteNet>
<CanvasParameter>
<T_min>0</T_min>
<T_max>5.0</T_max>
<F_min>0</F_min>
<F_max>20</F_max>
<State_min>0</State_min>
<State_max>1.0</State_max>
<Dense_min>0</Dense_min>
<Dense_max>2.5</Dense_max>
</CanvasParameter>
<CanvasNode Name="LIF0" />
</SimulationIO>
<SimulationRunParameter>
<max_iter>1000000</max_iter>
<t_begin>0</t_begin>
<t_end>{t_end}</t_end>
<t_report>{t_report}</t_report>
<t_state_report>{t_state_report}</t_state_report>
<t_step>{dt}</t_step>
<name_log>{basename}.log</name_log>
</SimulationRunParameter>
</Simulation>""".format(**{**self.model_param, **self.miind_param})
    
        elif self.model_param['N_inp'] == 2:
            
            xml_skellet = """<Simulation>
<WeightType>DelayedConnection</WeightType>
<Algorithms>
<Algorithm type="MeshAlgorithm" name="LIF" modelfile="{basename}.model" >
<TimeStep>{dt}</TimeStep>
<MatrixFile>{mat0}</MatrixFile>
<MatrixFile>{mat1}</MatrixFile>
</Algorithm>
<Algorithm type="RateFunctor" name="ExtInp">
<expression>{r_ext_E}</expression>
</Algorithm>
<Algorithm type="RateFunctor" name="InhInp">
<expression>{r_ext_I}</expression>
</Algorithm>
</Algorithms>
<Nodes>
<Node algorithm="LIF" name="LIF0" type="EXCITATORY_DIRECT" />
<Node algorithm="ExtInp" name="ExcInp0" type="NEUTRAL" />
<Node algorithm="InhInp" name="InhInp0" type="NEUTRAL" />
</Nodes>
<Reporting>
<Density node="LIF0" t_start="0.0" t_end="{t_end}" t_interval="{t_report}" />
<Rate node="LIF0" />
<!-- Display node="LIF0" -->
</Reporting>
<Connections>
<Connection In="ExcInp0" Out="LIF0">{KE} {hE} 0</Connection>
<Connection In="InhInp0" Out="LIF0">{KI} {hI} 0</Connection>
</Connections>
<SimulationIO>
<SimulationName>{basename}</SimulationName>
<OnScreen>FALSE</OnScreen>
<WithState>TRUE</WithState>
<WriteNet>FALSE</WriteNet>
<CanvasParameter>
<T_min>0</T_min>
<T_max>5.0</T_max>
<F_min>0</F_min>
<F_max>20</F_max>
<State_min>0</State_min>
<State_max>1.0</State_max>
<Dense_min>0</Dense_min>
<Dense_max>2.5</Dense_max>
</CanvasParameter>
<CanvasNode Name="LIF0" />
</SimulationIO>
<SimulationRunParameter>
<max_iter>1000000</max_iter>
<t_begin>0</t_begin>
<t_end>{t_end}</t_end>
<t_report>{t_report}</t_report>
<t_state_report>{t_state_report}</t_state_report>
<t_step>{dt}</t_step>
<name_log>{basename}.log</name_log>
</SimulationRunParameter>
</Simulation>""".format(**{**self.model_param, **self.miind_param})
        
        with open(self.xml_file, 'w+') as f:
            f.write(xml_skellet)
            
        return

    def get_density_dict(self):
                     
        marginal = self.sim.getMarginalByNodeName(self.node)        
                
        # load only each nth file
        #n = int(self.parameter['t_state_report']/self.parameter['dt'])
        data = marginal.density_PIF()
        
        density = {}
        density['t'] = data['times']
                
        bins = data['bins_v']
        dv = bins[1] - bins[0]  
        density['rho'] = data['v']
                
        density['v']    = bins + dv/2        
        # include rightmost edge 
        density['bins'] = np.append(bins, bins[-1] + dv)   
        
        self.bins = density['bins']
                    
        return density

    def get_rate_dict(self):
    
        n = int(self.miind_param['t_report']/self.miind_param['dt'])

        rates = self.sim.rates_report(n)                   
        node_index = self.sim.getIndexFromNode(self.node)
        
        rate = {}
        rate['t'] = self.miind_param['dt']*np.array(rates['times'])      
        rate['r'] = rates[node_index]
        
        return rate


class LIF_Response_EB():
    '''Estimate repsonse of LIF neuron to Poisson input using a event based simulation'''
    

    def __init__(self, model_param):
        
            
        self.model_param = model_param
        
        self.tau  = model_param['tau']
        self.v_th = model_param['v_th']
        self.v_r  = model_param['v_reset']

        self.hE = model_param['hE']

        self.rE    = model_param['KE']*model_param['r_ext_E']
        
        if model_param['N_inp'] == 2: 
            
            self.hI = model_param['hI']
            self.rI = model_param['KI']*model_param['r_ext_I']
            self.h_arr = [self.hE, self.hI]
                                
    def generate_spike_trains(self, T):
        '''
        Generate N spike trains
        :param: r: rate in Hz
        :param: T: simulation time s
        :param: N: number of trials    
        '''
        
        if self.model_param['N_inp'] == 2:
                        
            nE = np.random.poisson(self.rE*T) # draw number of spikes
            spike_time_arr_E = np.random.uniform(0, T, nE) # generate spike times
        
            marker_E = np.zeros(nE, dtype = np.int)
                    
            nI = np.random.poisson(self.rI*T) # draw number of spikes
            spike_time_arr_I = np.random.uniform(0, T, nI) # generate spike times
            
            marker_I = np.ones(nI, dtype = np.int)
                
            spike_times = np.concatenate((spike_time_arr_E, spike_time_arr_I))
            spike_type  = np.concatenate((marker_E, marker_I))
            
            idx_arr = np.argsort(spike_times)
            
            return spike_times[idx_arr], spike_type[idx_arr]
                              
        return

    def run_trials(self, N, T, v0 = None):
        
        if v0 is None:
            self.v0 = self.v_r
                        
        # membrane potential and output spikes for different trials
        v_mat = [] 
        os_mat = []

        for i in range(N):
            
            input_spikes, spike_types = self.generate_spike_trains(T)
            
            output_spikes, v = self.v_eb(input_spikes, spike_types)
            os_mat.append(output_spikes)
            v_mat.append(v)
            
        # caculate rate        
        os_arr = np.concatenate(os_mat)
        dt = 0.001 # resolution 
        
        # bins including the rightmost bin
        bins = np.arange(0, T+dt-1e-10, dt)
        
        spike_count, _ = np.histogram(os_arr, bins)
        rate = spike_count/dt/N
        
        t = bins[:-1] + 0.5*dt
                      
        return rate, t
            
    def v_eb(self, input_spikes, spike_types):
        
        t_ls = 0.0 # time last spike has arrived
        v_ls = self.v0 # membrane pontential at last spike arrival
                
        output_spikes = []
        v_arr = np.zeros(len(input_spikes))
                
        for i, (t_ns, st) in enumerate(zip(input_spikes, spike_types)): 
            
            #t_ns is the time of the next spike arrival
            dt = t_ns - t_ls 
            
            t_ls = t_ns
                        
            v_ls = v_ls*np.exp(-dt/self.tau) + self.h_arr[st]
        
            if v_ls > self.v_th:
                output_spikes.append(t_ls)
                v_ls = self.v_r
            
            v_arr[i] = v_ls
        
        return output_spikes, v_arr    
    
    
         
      
class LIF_Response_MC():
    '''Estimates response of a LIF neuron to a Poisson input using Monte Carlo simulations'''
    
    def __init__(self, model_param, mc_param):
        '''
        :param dict param: neuron and simulation model_param'''
                
        self.model_param = model_param  
        self.mc_param    = mc_param
                                            
        return

    def simulate(self):
        '''Simulate neuron response to N different randomly generated incoming Poisson spike trains'''
                                     
        self.euler()                
        
        return
    
    def save(self, h5f):
            
        density = self.get_density_dict()
        rate    = self.get_rate_dict()
                              
        # save density
        for key, value in density.items():
             
            path = 'mc/density/' + key
            h5f.create_dataset(path, data = value)
        
        # save rate
        for key, value in rate.items():
             
            path = 'mc/rate/' + key
            h5f.create_dataset(path, data = value)
            

        # save parameter                            
        h5f.create_group('mc/parameter')        
        h5f['mc/parameter'].update(self.mc_param)
                                            
        return

    def get_density_dict(self):

        self.calculate_density()
        
        density = {}
        density['t']      = self.t_report
        density['rho']    = self.rho_mat
        
        dv = (self.bins_v[1] - self.bins_v[0])/2        
        density['v'] = self.bins_v[:-1] + dv
        density['bins'] = self.bins_v
                            
        return density
        
    def get_rate_dict(self):

        self.calculate_rate()        
        
        rate = {}
        rate['t'] = self.t_report
        rate['r'] = self.r
        
        return rate
                            
    def calculate_density(self):
                
        bins_v = self.mc_param['bins_v']
                
        self.rho_mat = np.zeros((self.M, len(bins_v) - 1))
                
        for i, v in enumerate(self.v_mat.T):
            
            rho_t, _ = np.histogram(v, bins_v, density = True)
            
            self.rho_mat[i, :] = rho_t
        
        self.bins_v = bins_v 
                    
        return self.rho_mat
                     
    def calculate_rate(self):
        '''Caclualte smooth rate curve for each t_report'''
        
        dt = self.mc_param['dt']
        N_trial = self.mc_param['N_trial']
      
        # convole rate                                     
        box_kernel = np.ones(self.n)
                                        
        rate = np.convolve(self.spike_count, box_kernel, mode = 'same')/(dt*self.n)
        self.r  = rate[::self.n]/N_trial
        
        return 

    def euler(self):
        '''Simulate time evolution of neuron using euler'''
        
        # simulation parameter
        N_trial = self.mc_param['N_trial']
        dt = self.mc_param['dt']
                        
        # neuron model      
        tau     = self.model_param['tau'] # membrane time constant        
        v_th    = self.model_param['v_th'] # treshold
        v_rev   = self.model_param['v_rev'] # reversal potential
        v_min   = self.model_param['v_min'] # minimum potential
        v_reset = self.model_param['v_reset'] # reset potential

        # Poisson input
        hE        = self.model_param['hE'] # efficacy
        r_ext_E   = self.model_param['r_ext_E'] # input rate
        KE        = self.model_param['KE'] # number of indegrees 
        
        if self.model_param['N_inp'] == 1:
            
            def I():
                return hE*np.random.poisson(KE*r_ext_E*dt, size = N_trial)
                
        elif self.model_param['N_inp'] == 2:
            
            hI       = self.model_param['hI'] # efficacy
            r_ext_I  = self.model_param['r_ext_I'] # input rate
            KI       = self.model_param['KI'] # number of indegrees 

            def I():
                return hE*np.random.poisson(KE*r_ext_E*dt, size = N_trial) + hI*np.random.poisson(KI*r_ext_I*dt, size = N_trial)
                
        # simulation   
        t_report = self.mc_param['t_report']
        t_end    = self.mc_param['t_end']

        # number of steps
        self.N = int(np.floor(100*t_end/(100*dt)))
        # report only each nth euler loop
        self.n = int((t_report/dt))
        # number of recordings
        self.M = int(self.N/self.n)
        
        # membrane potential
        self.v_mat   = np.zeros((N_trial, self.M))                                    

        self.spike_count = np.zeros(self.N)
        self.t = dt*np.arange(1, self.N + 1)
        self.t_report = self.t[::self.n]
                
        # Initial value
        v0 = 0.0
        v = v0*np.ones(N_trial)
                
        k = 0
    
        for i in range(self.N):
#            
            # euler step
            dv = -(v - v_rev)/tau*dt 
            v = v + dv + I()

            # report spikes
            self.spike_count[i] = np.sum(v > v_th)
                    
            # reset                         
            v[v > v_th]  = v_reset        
            # reflecting barrier
            v[v < v_min] = v_min
            
            # report only each nth step
            if i % self.n == 0:
                self.v_mat[:, k] = v
                k += 1
                                
        return
    
class LIF_Theory():
    
    def __init__(self, model_param = None, diffusion_param = None):

        if model_param is not None:            
            self.model_param = model_param
            self.diffusion_approximation()
        elif diffusion_param is not None:
            self.mu   = diffusion_param['mu']
            self.sig  = diffusion_param['sig']
            self.sig2 = self.sig**2
            self.v_th = diffusion_param['v_th']
            self.v_r = diffusion_param['v_r']
            
            if 'tau' in diffusion_param:            
                self.tau = diffusion_param['tau']
            
            self.r0 = None
            
        self.variable_trafo()
        self.lam_cache = {}
        self.coeff_cache = {}
        
        # set precision
        mp.mp.dps = 100

    
    def variable_trafo(self):
        '''Fokker-Planck equation be simplifite by variable transformation'''
        
        self.x_th = np.sqrt(2)*(self.v_th - self.mu)/self.sig
        self.x_r  = np.sqrt(2)*(self.v_r - self.mu)/self.sig
        
        # similarity trafo f transform FP eq. into Schroedinger eq.
        self.f_th = mp.exp(-0.25*self.x_th**2)
        self.f_r  = mp.exp(-0.25*self.x_r**2)
        
        self.y_th = -self.x_th
        self.y_r  = -self.x_r

        return
    
    def diffusion_approximation(self):
                                
        h_arr     = self.model_param['h_arr'] # efficacy               
        tau       = self.model_param['tau'] # membrane time constant
        r_ext_arr = self.model_param['r_ext_arr'] # external input rate  
        K_arr     = self.model_param['K_arr'] # number of indegrees 

        self.v_th   = self.model_param['v_th']        
        self.v_r    = self.model_param['v_reset']        
        self.v_min  = self.model_param['v_reset']                
        self.tau    = self.model_param['tau']

        self.mu   = tau*np.sum(K_arr*h_arr*r_ext_arr)
        self.sig2 = tau*np.sum(K_arr*h_arr**2*r_ext_arr)
        self.sig  = np.sqrt(self.sig2)
                                    
        if 'tau_ref' in self.model_param:
            self.tau_ref = self.model_param['tau_ref']
        else:
            self.tau_ref = 0.0
            
        self.r0 = None

    def get_r0(self):
        '''Stationary rate'''
        
        if not self.r0 is None:
            return self.r0
            
        a = (self.v_r  - self.mu)/self.sig
        b = (self.v_th - self.mu)/self.sig
                                
        r0 = 1./(self.tau*mp.sqrt(mp.pi)*mp.quad(lambda x: (1.+mp.erf(x))*mp.exp(x**2), [a, b]))
    
        if self.mu > self.v_th:
            self.r0_approx = 1./(self.tau*np.log(self.mu/(self.mu - self.v_th)))
                
        self.r0 = float(r0)
                        
        return self.r0
        
    def get_phi_0(self, v_arr = None):
        '''Stationary distribution
        :param arr v_arr: membrane potential'''
        
        if self.r0 is None:
            self.get_r0()
        
        if v_arr is None:
            v_arr = np.linspace(self.v_min, self.v_th, 100)
        
        # coordiante transformation
        x_arr = np.sqrt(2)*(v_arr - self.mu)/self.sig

        x_th = mp.sqrt(2.)*(self.v_th - self.mu)/self.sig
        x_r  = mp.sqrt(2.)*(self.v_r - self.mu)/self.sig

        self.phi_0 = np.zeros(len(x_arr))
                    
        for i, x in enumerate(x_arr):
            if x < x_r:   
                a = x_r
            else:
                a = x
                                    
            self.phi_0[i] = self.tau*self.r0*mp.exp(-0.5*x**2)*mp.quad(lambda y: mp.exp(0.5*y**2), [a, x_th])
        
        # back transformation
        self.phi_0 = np.sqrt(2)/self.sig*self.phi_0
                    
        return self.phi_0
                  
    def get_lam(self, n, Type):
        '''Get eigenvalues
        :param int n: nth eigenvalue
        :param type str: either real or complex 
        '''

        data_path = './data/eigenvalues/'
        filename  = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.h5'
                
        if os.path.isfile(data_path + filename):
        
            with h5.File(data_path + filename, mode = 'r') as hf5:                                    
                if Type == 'real':
                    if hf5[Type].attrs['N'] > n:                    
                        return hf5[Type][n]                  
                elif Type == 'complex':
                    if hf5[Type].attrs['N'] >= n - 1:                    
                        return hf5[Type][n - 1]  

        print('Eigenvalues have not been caculated yet for \n'
              f'mu: {self.mu:.2f} \n'
              f'sig: {self.sig:.2f}')
        
        return

    def EV_cxroot(self):
        
        import cxroots
        
        L = 5

        f_th = mp.exp(-0.25*self.x_th**2)
        f_r  = mp.exp(-0.25*self.x_r**2)
        
        f  = lambda z: np.complex(mp.pcfu(z, self.x_th)/f_th - mp.pcfu(z, self.x_r)/f_r)
                
#         C = cxroots.Rectangle([-L, 0], [-0.1, L])
        
        C = cxroots.Rectangle([-L, 0], [-0.1, 0.1])
        
        r = C.roots(f)
        r.show()
        
        return

    def get_coefficients(self, n, Type):
        
        if n in self.coeff_cache:
            return self.coeff_cache[n]
        
        lam = self.get_lam(n, Type)

        z = -0.5 + lam

        f_th = mp.exp(-0.25*self.y_th**2)
        f_r  = mp.exp(-0.25*self.y_r**2)
                
        a =  mp.pcfu(z, self.y_th)/f_th
        b = -mp.pcfv(z, self.y_th)/f_th                
        c =  mp.pcfv(z, self.y_r)/f_r - mp.pcfv(z, self.y_th)/f_th
        
        self.coeff_cache[n] = [a,b,c]
        
        return self.coeff_cache[n]
        
    def phi_n(self, n, Type, v):
        
        # eigenvalue
        lam = self.get_lam(n, Type)
        z = lam -0.5        
                
        # coefficients 
        a,b,c = self.get_coefficients(n, Type)
        
        if Type == 'real':
            phi_arr = np.zeros_like(v)
        elif Type == 'complex':
            phi_arr = np.zeros_like(v, dtype = np.complex)

        v_arr = v[v < self.v_r]
        
        y_arr = -np.sqrt(2)*(v_arr - self.mu)/self.sig
                
        for i, y in enumerate(y_arr):
            
            phi_arr[i] = c*mp.exp(-0.25*y**2)*mp.pcfu(z, y)

        v_arr = v[v >= self.v_r]
        y_arr = -np.sqrt(2)*(v_arr - self.mu)/self.sig

        k = i+1

        for y in y_arr:
            phi_arr[k] = mp.exp(-0.25*y**2)*(a*mp.pcfv(z, y) + b*mp.pcfu(z, y))
            k += 1
       
        return phi_arr             
    
    def dphi_n(self, n, Type, v):

        # coefficients 
        lam = self.get_lam(n, Type)
        z = lam - 0.5

        a,b,c = self.get_coefficients(n, Type)
        
        dphi_n = np.zeros_like(v, dtype = np.complex)

        v_arr = v[v < self.v_r]
        
        y_arr = -np.sqrt(2)*(v_arr - self.mu)/self.sig
                        
        for i, y in enumerate(y_arr):
            
            dphi_n[i] = -c*mp.exp(-0.25*y**2)*mp.pcfu(z-1, y)

        v_arr = v[v >= self.v_r]
        y_arr = -np.sqrt(2)*(v_arr - self.mu)/self.sig

        k = i+1

        for y in y_arr:
            dphi_n[k] = -mp.exp(-0.25*y**2)*(a*mp.rf(0.5 - z, 1)*mp.pcfv(z-1, y) + b*mp.pcfu(z-1, y))
            k += 1
       
        return dphi_n 
    
    def get_b_tilde(self, n, Type):
    
        # eigenvalue 
        lam = self.get_lam(n, Type)
        z = lam - 0.5
        
        # coefficients for phi
        a,b,c = self.get_coefficients(n, Type)

        # integrand for inner product 
        def phi_times_phi_tilde(x):
                
            if x < self.x_r:
                return mp.pcfu(z, -x)*c*mp.pcfu(z, -x)
            else:
                return mp.pcfu(z, -x)*(a*mp.pcfv(z, -x) + b*mp.pcfu(z, -x))
            
        # inner product
        inv_b = mp.quad(phi_times_phi_tilde, [-mp.inf, self.x_r, self.x_th])

        return 1./inv_b
    
    
    def phi_tilde_n(self, n, Type, v):
        '''Adjoint eigenfuncti'''

        # eigenvalue 
        lam = self.get_lam(n, Type)
        z = lam - 0.5

        b_tilde = self.get_b_tilde(self, n, Type)
            
        x_arr = np.sqrt(2)*(v-self.mu)/self.sig

        phi_tilde = np.zeros_like(x_arr, dtype = np.complex128)

        for i, x in enumerate(x_arr):

            phi_tilde[i] = b_tilde*mp.exp(0.25*x**2)*mp.pcfu(z, -x)
                                                
        return phi_tilde
    
    def c0_n(self, n, Type, v0):
        '''Initial weighting coefficent assuming all neurons are at located at v0 at t=0
        which can expressed in terms of a delta-distribution'''

        # coefficients 
        lam = self.get_lam(n, Type)
        z = lam - 0.5
        
        b_tilde = self.get_b_tilde(n, Type)
      
        x0 = np.sqrt(2.)*(v0 - self.mu)/self.sig            
        
        return b_tilde*mp.exp(0.25*x0**2)*mp.pcfu(z, -x0)

    def sum_r_n(self, k, t):
        '''Rate dynamics due to modes up to order k'''
        
        sum_r = np.zeros(len(t))
        
        for n in range(k):
        
            sum_r += self.r_n(n, t)
            
        return sum_r
        
    def r_n(self, n, Type, t, v0 = None):
        '''Rate dynamics due to mode n'''
        
        s = t/self.tau
        
        lam = self.get_lam(n, Type)
        
        if v0 is None:
            v0 = self.v_r
        
        c0 = np.complex(self.c0_n(n, Type, v0))
        
        return np.sqrt(2./np.pi)*2.*(c0*np.exp(lam*s)).real
                
    def mode_n(self, n, Type, t, v0 = None):
        
        pass

        
    def EV_save_complex(self):
        '''Save eigenvalues to file'''

        filename = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.h5'        
        data_path = './data/eigenvalues/'        
        
        # add eigenvalues to old file if its already exitsts 
        with h5.File(data_path + filename, mode = 'a') as hf5:                                            
            if 'complex' in hf5:                
                N = hf5['complex'].attrs['N']
                M = len(self.lam_complex)
                hf5['complex'].resize((N+M,))
                hf5['complex ceq'].resize((N+M,))                
                hf5['complex'][N:] = self.lam_complex
                hf5['complex ceq'][N:] =  self.deter_complex                        
                hf5['complex'].attrs['N'] = N + M                                        
            else: 
                N = len(self.lam_complex)                                   
                dset = hf5.create_dataset('complex', (N,), maxshape=(1000,), dtype = np.complex)
                dset[:N] = self.lam_complex
                dset.attrs['min_real'] = self.min_real                
                dset.attrs['N'] = N                                        
                dset = hf5.create_dataset('complex ceq', (N,), maxshape=(1000,), dtype = np.complex)
                dset[:N] = self.deter_complex  
            if not 'mu' in hf5.attrs.keys():     
                hf5.attrs['mu'] = self.mu
                hf5.attrs['sig'] = self.sig
                
                    
    def EV_save_real(self):
        '''Save eigenvalues to file'''

        filename = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.h5'        
        data_path = './data/eigenvalues/'        
        
        # add eigenvalues to old file if its already exitsts 

        with h5.File(data_path + filename, mode = 'a') as hf5:                                            
            # save eigenvalues                                
            if 'real' in hf5:                
                N = hf5['real'].attrs['N']
                M = len(self.lam_real)
                if M > 0:
                    hf5['real'].resize((N+M,))
                    hf5['real'][N:] = self.lam_real
                    hf5['real ceq'][N:] =  self.deter_real    
                    hf5['real'].attrs['N'] = N + M                                                                                
            # create new file if it does not exist
            else:
                N = len(self.lam_real) + 1                                   
                dset = hf5.create_dataset('real', (N,), maxshape=(1000,))
                dset[1:N] = self.lam_real                    
                dset.attrs['N'] = N
                dset.attrs['min_real'] = self.min_real
                dset = hf5.create_dataset('real ceq', (N,), maxshape=(1000,))
                dset[1:N] = self.deter_real
            if not 'mu' in hf5.attrs.keys():     
                hf5.attrs['mu'] = self.mu
                hf5.attrs['sig'] = self.sig
        return 

    
    def EV_check_real(self):
        
        filename = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.h5'        
        data_path = './data/eigenvalues/'        
        files = [f for f in os.listdir(data_path) if f.endswith('.h5')]
        
        done = False
        
        if any(filename in f for f in files):
            with h5.File(data_path + filename, mode = 'r') as hf5:                                                        
                
                if 'real' in hf5:
                    if hf5['real'].attrs['min_real'] < self.min_real:                                        
                        print('Real eigenvalues have already been caculated for \n'
                              f'mu: {self.mu:.2f} \n'
                              f'sig: {self.sig:.2f} \n'                          
                              f'up to min_real: {hf5["real"].attrs["min_real"]}')
                        done = True
                    else: 
                        print('Eigenvalues have been already caculated for \n'
                              f'mu: {self.mu:.2f} \n'
                              f'sig: {self.sig:.2f} \n'                           
                              f'up to min_lam_real: {hf5["real"].attrs["min_real"]} \n'
                              f'looking for new ones from {hf5["real"].attrs["min_real"]} to {self.min_real}')
                        self.max_real = hf5['real'].attrs['min_real']
        
        return done
        
    def EV_check_complex(self):

        filename = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.h5'        
        data_path = './data/eigenvalues/'                        
        files = [f for f in os.listdir(data_path) if f.endswith('.h5')]

        done = False
                                
        if any(filename in f for f in files):
            with h5.File(data_path + filename, mode = 'r') as hf5:             
                if 'complex' in hf5:         
                    lam = hf5['complex'][-1]                                                                   
                    if lam.real < self.min_real:
                        done = True
                    if lam.real < self.max_real:
                        self.max_real = lam.real                                                            
        
        return done
                                
    def EV_brute_force(self, 
                       min_real, 
                       max_real = -0.1,
                       min_imag = 0,
                       max_imag = 0, 
                       real = True,
                       comp = True,
                       ev_plot = False):
    
        self.min_real = min_real
        self.max_real = max_real        
        self.max_imag = max_imag
        self.min_imag = min_imag
        self.ev_plot     = ev_plot
        
        if real:     
            if not self.EV_check_r():                            
                self.EV_real()        
                self.EV_save_real()
        if comp:
            if not self.EV_check_complex():
                self.EV_complex()
                self.EV_save_complex()
                
        return
            
    def EV_real(self):
        ''' Determine eigenvalues on the real line'''
                
        # Eigenfunctions are given by a linear combination of parabolic cylinder functions 
        # The three coefficients which need to chosen such that the eigenfunctions fulfil the bc 
        # This leads to homogeneous matrix equation for the coefficients
        # We look for those ev for which the determinant of the matirx is zero 
        f_th = mp.exp(-0.25*self.y_th**2)
        f_r  = mp.exp(-0.25*self.y_r**2)
                                            
        deter = lambda z: mp.pcfu(z, self.y_th)*f_r - mp.pcfu(z, self.y_r)*f_th
                          
        #=======================================================================
        # Zeros on the real line
        #=======================================================================        
        Nz = 200*int(-self.min_real)
        
        z_arr = np.linspace(self.min_real, self.max_real, Nz) - 0.5

        deter_arr = np.zeros_like(z_arr)
         
        for i, z in enumerate(z_arr):
            
            deter_arr[i] = deter(z)
 
        # zero crossing idx                
        zc_idx = np.abs(np.diff(np.sign(deter_arr)))/2
        zc_idx = np.append(zc_idx, 0)
        zc_idx = zc_idx.astype(np.bool)
        
        a_arr = z_arr[zc_idx]
        b_arr = z_arr[np.roll(zc_idx, 1)]
         
        z0_arr = np.zeros(len(a_arr))
        deter0_arr = np.zeros(len(a_arr))
  
        for i, (a, b) in enumerate(zip(a_arr, b_arr)):
             
            z0_arr[i] = brentq(deter, a, b)
            deter0_arr[i] = deter(z0_arr[i])
        
        self.lam_real = z0_arr + 0.5
        # Swap order so that the first eigenvalue is the closest to zero
        self.lam_real   = self.lam_real[::-1]
        self.deter_real = deter0_arr[::-1]
                        
        return        
                                                
    def EV_complex(self):
        '''Brute force scans complex plane and searches for those eigenvalues for which
        are roots of the characteristic equation'''
          
        # Grid points on the real line
        Nx = 10*int(self.max_real-self.min_real) 
        # Grid points on the complex plance    
        Ny = 10*int(self.max_imag - self.min_imag)
        
        x_arr = np.linspace(self.min_real, self.max_real, Nx) - 0.5
        y_arr = np.linspace(self.min_imag, self.max_imag, Ny)

        # Characteristic equation
        f_th = mp.exp(-0.25*self.y_th**2)
        f_r  = mp.exp(-0.25*self.y_r**2)                                        
        deter = lambda z: mp.pcfu(z, self.y_th)*f_r - mp.pcfu(z, self.y_r)*f_th
        
        # Allocate matrices in which we save the crossings of 
        # the imaginary and real part of the characteristic eq.  
        zc_real_mat = np.zeros((Ny, Nx-1), dtype = np.float)
        zc_imag_mat = np.zeros((Ny, Nx-1), dtype = np.float)

        for j, y in enumerate(y_arr):             
            
            deter_arr = np.zeros(Nx, dtype = np.complex)
            
            for i, x in enumerate(x_arr):
                                
                z = complex(x, y) 
                deter_arr[i] = deter(z)
                
            zc_real_mat[j,:] = np.abs(np.diff(np.sign(deter_arr.real)))/2
            zc_imag_mat[j,:] = np.abs(np.diff(np.sign(deter_arr.imag)))/2

        sig = 1.5

        blur_zc_real_mat = gaussian_filter(zc_real_mat, sigma = sig, mode = 'mirror')
        blur_zc_imag_mat = gaussian_filter(zc_imag_mat, sigma = sig, mode = 'mirror')

        poles_mat = blur_zc_real_mat*blur_zc_imag_mat

        if self.ev_plot:
            gs  = plt.GridSpec(1, 2)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            
            ax0.imshow(zc_real_mat-zc_imag_mat)
            ax0.set_xlabel(u'Real($\lambda$)', fontsize = 18)
            ax0.set_ylabel(u'Imag($\lambda$)', fontsize = 18)
            
            ax1.imshow(poles_mat)
            ax1.set_xlabel(u'Real($\lambda$)', fontsize = 18)
            
            filename = f'EV_mu_{self.mu:.2f}_sig_{self.sig:.2f}.png'
            
            plt.savefig('./figures/eigenvalues/' + filename)
            
        # consider only candidates larger than threshold
        th = 0.1*np.max(poles_mat.flatten())                
        poles_mat[poles_mat < th] = 0
        
        X, Y = np.meshgrid(x_arr[:-1], y_arr)
    
        # candidate locations
        x_cand_arr = X[poles_mat > 0]
        y_cand_arr = Y[poles_mat > 0]
               
        r_cand_arr = np.column_stack((x_cand_arr, y_cand_arr))
               
        z0_cand_arr = self.cluster(r_cand_arr)        
        z0_arr = np.zeros(len(z0_cand_arr), dtype = np.complex)        
        deter0_arr = np.zeros(len(z0_cand_arr), dtype = np.complex)
        
        for i, z0 in enumerate(z0_cand_arr):
                                     
            for method in ['newton']:
                
                success = False
                                        
                try: 
                    z0_arr[i] = mp.findroot(deter, mp.mpc(z0[0], z0[1]), solver = method)
                    deter0_arr[i] = deter(z0_arr[i])                 
                except:                    
                    print(f'{method} failed at {z0}')
                    z0_arr[i] = np.nan
                else:
                    print(f'{method} succeded at {z0}')                                   
                    success = True
                    break
                
            if not success:
                    print(f'No sucess at {z0} save value anyway')
                    z0_arr[i] = mp.findroot(deter, mp.mpc(z0[0], z0[1]), solver = method, verify = False)
                    deter0_arr[i] = deter(z0_arr[i])                  
#                     tol = 1e-10
#                     z0 = mp.findroot(deter, mp.mpc(z0[0], z0[1]), solver = 'newton', verify = False)
#                     deter_0 = mp.fabs(deter(z0)) 
#                                                             
#                     eps = 1e-3
#                     
#                     z0_nh_arr = [z0 + eps, z0 - eps, z0 + mp.mpc(0, eps), z0 - mp.mpc(0, eps)]
#                     
#                     success = True
#                     for z in z0_nh_arr:
#                         
#                         if deter_0/mp.fabs(deter(z)) > tol:
#                             print('Problem with candidate for eigenvalue at \n'
#                                   f'{z0}')
#                             success = False
#                             break
#                     if success:
#                         z0_arr[i] = z0
                                                        
        self.lam_complex   = z0_arr + 0.5
        self.deter_complex = deter0_arr
        
        return

    def cluster(self, r_cand_arr):
        
        # threshold distance 
        cluster_arr = [r_cand_arr[0, :]]
                
        center_arr = [r_cand_arr[0, :]]
                    
        th = 1
                    
        for r in r_cand_arr[1:,:]:
            
            for i, center in enumerate(center_arr):
                
                new_cluster = True                               
                dist = np.linalg.norm(r - center)
                    
                if dist < th:
                    cluster_arr[i] = np.vstack((cluster_arr[i], r)) 
                    center_arr[i] = np.mean(cluster_arr[i], axis = 0)
                    new_cluster = False
                    break
            
            if new_cluster:                
                cluster_arr.append(r)
                center_arr.append(r)
                
        return center_arr
                        
    def EV(self, n, m = 1., method = 'bf'):
        '''Find numeric solution for eigenvalues determined by the characteristic equation'''
                
        if method == 'bf':
            self.EV_brute_force()
        elif method == 'cx_root':
            self.EV_cxroot()
        elif method == 'test':
            self.EV_test()
#         X = np.meshgrid(x)
#         Y = np.meshgrid(y)
#         
#         err = np.zeros_like(X)
#         
#         f_th = exp(0.25*self.x_th)
#         f_r  = exp(0.25*self.x_r)
#                     
#         for i, x in enumerate(X):
#             for j, y in enumerate(Y):
#             
#                 z = mpc(x, y)
#             
#                 err[j, i] = pcfu(self.x_th, z)/f_th - pcfu(self.x_r, z)/f_r
#                     
                     
        return
    






