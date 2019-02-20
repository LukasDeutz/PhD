'''
Created on 3 Jan 2019

@author: lukas
'''
import numpy as np
from mpmath import findroot, cosh, sinh, pcfu, pcfv, exp, mpc
import h5py as h5
from timeit import itertools
import miind_api as api
import mesh3

from miind_library.generate_simulation import generate_model, generate_empty_fid, generate_matrix

# from scipy.integrate import quad
# from scipy.special import erf

import mpmath as mp 
import nest

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
                        
        # Add extra cells add the end avoid that neurin mass gets kicked out of the strip
        self.v_max = self.v_th + 2*self.hE
              
        if self.v_min > self.v_reset:
            raise ValueError ("v_min must be less than v_reset.")

        with open(self.basename + '.mesh','w') as meshfile:
            meshfile.write('ignore\n')
            meshfile.write('{}\n'.format(self.dt))

            ts = self.dt * np.arange(self.N_grid)
            pos_vs = self.v_rev + (self.v_th - self.v_rev)*np.exp(-ts/self.tau)
            pos_vs = np.insert(pos_vs, 0, self.v_max)
            neg_vs = self.v_rev + (self.v_min - self.v_rev)*np.exp(-ts/self.tau)

            self.dv_stat = pos_vs[-1]

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

        v_plus = self.v_rev + self.dv_stat
        self.v_min  = self.v_rev - self.dv_stat

        with open(statname,'w') as statfile:
            statfile.write('<Stationary>\n')
            format = "%.9f"
            statfile.write('<Quadrilateral>')
            statfile.write('<vline>' +  str(self.v_min) + ' ' + str(self.v_min) + ' ' +  str(v_plus) + ' ' + str(v_plus) + '</vline>')
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
        h_arr     = self.model_param['h_arr'] # efficacy
        N_mc      = self.miind_param['N_mc'] # number of monte carlo samples
        uac       = self.miind_param['uac'] # caculate transition matrix geomtrically
        
        generate_model(self.basename, v_reset, v_th)
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
    
    def __init__(self, model_param):
            
        self.model_param = model_param
        self.diffusion_approximation()
        self.variable_trafo()
        self.lam_cache = {}
    
    def variable_trafo(self):
        '''Fokker-Planck equation be simplifite by variable transformation'''
        
        self.x_th = np.sqrt(2)*(self.v_th - self.mu)/self.sig
        self.x_r  = np.sqrt(2)*(self.v_r - self.mu)/self.sig
    
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
        
#         r0 = 1./(self.tau*mp.sqrt(2.*mp.pi))*mp.quad(lambda x: mp.exp(-0.5*x**2), [b, mp.inf])
        
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
                  
    def get_lam(self, n):
        '''Get eigenvalues '''
        
        if not self.lam_cache.has_key(n):            
            self.lam_cache[n] = self.EV(n)
        
        return self.lam_cache[n]

    def get_zeta(self, n):
        
        return self.theta/self.sig2*np.sqrt(self.eta**2 + 2*self.sig2*self.get_lam(n)) 
        
    def CEAS(self, n, m = 1):
        '''Calculates analytic approximation of the characteristic equation'''
            
        i = np.complex(0, 1)
        a_n = 2*np.pi*n
        
        if self.xi == 0: 
            
            zeta = i*2*np.pi*n    
        
        elif self.xi < 0: # noise-dominated regime: Eigenvalues are real, i.e. zeta must be purely imaginary or real                
            
            zeta = i * (a_n + 1./a_n * (1. + self.xi - np.exp(self.xi) + (-1)**m * np.sqrt((1 + self.xi - np.exp(self.xi))**2 - 2.*a_n**2*(np.exp(self.xi) - 1))))                 
        
        else: # drift-dominated regime: Eigenvalues are complex, i.e. zeta must have a nonzero real and complex part        
            
            zeta = self.xi + np.log(1 + np.sqrt(1 - np.exp(-2*self.xi))) + (-1)**m*i*a_n
            
        return zeta
            
    def CES(self, n, m = 1):
        '''Finds numeric solution for characteristic equation for given n and xi'''
            
        # characteristic equation
        ce = lambda zeta: zeta*cosh(zeta) + self.xi*sinh(zeta) - np.exp(self.xi)*zeta
        # find root using analytic approximation as the start point
                    
        zeta = findroot(ce, self.CEAS(n, m), maxsteps= 100) 
            
        return zeta
    
    def EV_brute_force(self):
        '''Find numeric solution for eigenvalues determined by the characteristic equation brute force'''
        
        # Brute Force scan complex plane    
        x_arr = np.linspace(-5, 0, 100)
        y_arr = np.linspace(0, 5, 100)

        X,Y = np.meshgrid(x_arr, y_arr)
        
        # Eigenfunctions are given by a linear combination of parabolic cylinder functions 
        # The three coefficients which need to chosen such that the eigenfunctions fulfil the bc 
        # This leads to homogeneous matrix equation for the coefficients
        # We look for those ev for which the determinant of the matirx is zero 
        deter = np.zeros_like(x_arr)
        
        f_th = exp(-0.25*self.x_th)
        f_r  = exp(-0.25*self.x_r)
        
        for i, x in enumerate(x_arr):
            
            deter[i] = f_th*pcfu(z_n, self.x_th) - f_r*pcfu(z_n, self.x_r) 
        
                                
#         for i, x,y in enumerate(zip(X.flatten(), Y.flatten())):
#               
#             lam_n = np.complex(x, y)
#             z_n   = 1. - lam_n              
#             err_arr[i] = pcfu(z_n, x_reset) - pcfu(z_n, x_th)
#               
#         err_mat = np.reshape(err_arr, (L, L))
        
        pass
                
        return
        
            
    def EV(self, n, m = 1. method = 'bf'):
        '''Find numeric solution for eigenvalues determined by the characteristic equation'''
        
        if method == 'bf'
            self.EV_brute_force()
        
        
        
        
        X = np.meshgrid(x)
        Y = np.meshgrid(y)
        
        err = np.zeros_like(X)
        
        f_th = exp(0.25*self.x_th)
        f_r  = exp(0.25*self.x_r)
                    
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
            
                z = mpc(x, y)
            
                err[j, i] = pcfu(self.x_th, z)/f_th - pcfu(self.x_r, z)/f_r
                    
                     
        return
                     
              
    def a_0(self, n):
        '''Initial value of the expansion coefficients assuming density is given by a delta distribution a v=0 at t=0''' 
        
        pass
        
    
    def sum_r_n(self, k, t):
        '''Rate dynamics due to modes up to order k'''
        
        sum_r = np.zeros(len(t))
        
        for n in range(k):
        
            sum_r += self.r_n(n, t)
            
        return sum_r
        
    def r_n(self, n, t):
        '''Rate dynamics due to mode n'''
        
        pass 
    
    def flux_n(self, n):
                
        pass
                
    def phi_0(self, v):
        '''Stationary solution of the Fokker-Planck operator'''

        pass
        
    def phi_n(self, n, v):
        '''Eigenfunction corresponding to lam_n'''
    
        pass





