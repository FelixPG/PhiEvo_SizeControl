""" initialization script, import and copy data to like named variables in other modules.
"""

# ranges over which parmeters in classes_eds2 classes can vary. Defined in mutation.py
# NB use 1.0 etc to keep params real not int.

T=0.5 #typical time scale
C=1.5 #typical concentration
L=1.0 #typical size for diffusion

#dictionary key is the Class.attribute whose range is give an a real number or as a LIST
# indicating the interval over which it has to vary

dictionary_ranges={}
dictionary_ranges['Species.degradation']=1.0/T
#dictionary_ranges['Species.concentration']=C   # for certain species, eg kinases that are not trans
dictionary_ranges['Species.diffusion']=0   # for ligands diffusion
dictionary_ranges['TModule.rate']=2.0*C/T
dictionary_ranges['TModule.basal']=1.0
dictionary_ranges['CorePromoter.delay']=0   # convert to int(T/dt) in run_evolution.py
dictionary_ranges['TFHill.hill']=10.0
dictionary_ranges['TFHill.threshold']=2*C
dictionary_ranges['PPI.association']=2.0/(C*T)
dictionary_ranges['PPI.disassociation']=2.0/T
dictionary_ranges['Phosphorylation.rate']=1.0/T
dictionary_ranges['Phosphorylation.hill']=5.0
dictionary_ranges['Phosphorylation.threshold']=C
dictionary_ranges['Phosphorylation.dephosphorylation']=1.0/T
dictionary_ranges["Degradation.rate"] = 0.0

#################################################################################
# names of c-code files needed by deriv2.py (. for extension only)
# normally need 'header', 'utilities' (loaded before python derived routines and
# ['fitness', 'geometry', 'init_history', 'input', 'integrator', 'main' ] placed afterwards
# skip by setting cfile[] = ' ' or ''

cfile = {}
cfile['utilities'] = 'utilities_modif.c'
cfile['fitness'] = 'fitness_sizecontrol_pareto.c'
cfile['init_history'] = 'init_history_sizecontrol.c'
cfile['input'] =  'input_sizecontrol.c'
cfile['integrator'] =  'euler_integrator2.c'
cfile['main'] =  'main_general_modif.c'

#################################################################################
# mutation rates
dictionary_mutation={}

# Rates for nodes to add
dictionary_mutation['random_gene()']=0.01
dictionary_mutation['random_gene(\'TF\')']=0.01
dictionary_mutation['random_gene(\'Kinase\')']=0#0.01
dictionary_mutation['random_gene(\'Ligand\')']=0.00
dictionary_mutation['random_gene(\'Receptor\')']=0.00


dictionary_mutation['random_Interaction(\'TFHill\')']=0.002
dictionary_mutation['random_Interaction(\'PPI\')']=0.002
dictionary_mutation['random_Interaction(\'Phosphorylation\')']=0#0.002

# Rates for nodes to remove
dictionary_mutation['remove_Interaction(\'TFHill\')']=0.005
dictionary_mutation['remove_Interaction(\'PPI\')']=0.005
dictionary_mutation['remove_Interaction(\'CorePromoter\')']=0.01
dictionary_mutation['remove_Interaction(\'Phosphorylation\')']=0#0.005

# Rates to change parameters for a node
dictionary_mutation['mutate_Node(\'Species\')']=0.1
dictionary_mutation['mutate_Node(\'TFHill\')']=0.1
dictionary_mutation['mutate_Node(\'CorePromoter\')']=0.1
dictionary_mutation['mutate_Node(\'TModule\')']=0.1
dictionary_mutation['mutate_Node(\'PPI\')']=0.1
dictionary_mutation['mutate_Node(\'Phosphorylation\')']=0#0.1

#rates to change output tags.  See list_types_output array below
dictionary_mutation['random_add_output()']=0.0
dictionary_mutation['random_remove_output()']=0.0
dictionary_mutation['random_change_output()']=0.0


#############################################################################
# parameters in various modules, created as one dict, so that can then be passed as argument

# Needed in deriv2.py to create C program from network
prmt = {}
prmt['nstep'] = 50000         #number of time steps
prmt['ncelltot']=1            #number of cells in an organism
prmt['nneighbor'] = 3 # must be >0, whatever geometry requires, even for ncelltot=1
prmt['ntries'] = 10    # number of initial conditions tried in C programs
prmt['dt'] = 0.05    # time step


# Generic parameters, transmitted to C as list or dictionary.
# dict encoded in C as: #define KEY value (KEY converted to CAPS)
# list encoded in C as: static double free_prmt[] = {}
# The dict is more readable, but no way to test in C if given key supplied.  So
# always include in C: #define NFREE_PRMT int. if(NFREE_PRMT) then use free_prmt[n]
# prmt['free_prmt'] = { 'step_egf_off':20 }  # beware of int vs double type
# prmt['free_prmt'] = [1,2]



# Needed in evolution_gill to define evol algorithm and create initial network
prmt['npopulation'] =50
prmt['ngeneration'] =501
prmt['tgeneration']=1.0       #initial generation time (for gillespie), reset during evolution
prmt['noutput']=1    # to define initial network
prmt['ninput']=2
prmt['freq_stat'] = 5     # print stats every freq_stat generations
prmt['frac_mutate'] = 0.5 #fraction of networks to mutate
prmt['redo'] = 1   # rerun the networks that do not change to compute fitness for different IC

# used in run_evolution,
prmt['nseed'] = 30  # number of times entire evol procedure repeated, see main program.
prmt['firstseed'] = 0  #first seed


prmt['multipro_level']=1
prmt['pareto']=1
prmt['npareto_functions'] = 2
prmt['rshare']= 0.0
prmt['plot']=0
prmt['langevin_noise']=0

## prmt['restart']['kgeneration'] tells phievo to save a complete generation
## at a given frequency in a restart_file. The algorithm can relaunched at
## backuped generation by turning  prmt['restart']['activated'] = True and
## setting the generation and the seed. If the two latters are not set, the
## algorithm restart from the highest generation  in the highest seed.
prmt['restart'] = {}
prmt['restart']['activated'] = False #indicate if you want to restart or not
prmt['restart']['freq'] = 50  # save population every freq generations
#prmt['restart']['seed'] =  0 # the directory of the population you want to restart from
#prmt['restart']['kgeneration'] = 50  # restart from After this generation number
#prmt['restart']['same_seed'] = True # Backup the random generator to the same seed

list_unremovable=['Input','Output']

# control the types of output species, default 'Species' set in classes_eds2.    This list targets
# effects of dictionary_mutation[random*output] events to certain types, when we let output tags
# move around
#
list_types_output=['TF']

# necessary imports to define following functions
import random
from phievo.Networks import mutation

# Two optional functions to add input species or output genes, with IO index starting from 0.
# Will overwrite default variants in evol_gillespie if supplied below

def init_network():
   seed=int(random.random()*100000)
   g=random.Random(seed)
   net=mutation.Mutable_Network(g)
   s_d, tm_d, prom_d = {},{},{}

   net.activator_required=0
   net.fixed_activity_for_TF=True
  
   param = [["Input",0]]
  # Input0 is the VOLUME
   s_d[0] = net.new_Species(param)
   param = [["TF",1],["Input",1]]
  # Input1 is the SWITCH
   s_d[1] = net.new_Species(param)
  # Output 0 is the START decision maker
   param = [["TF",0],["Degradable",0.083827],["Complexable"],['Output',0]]
   tm_d[2],prom_d[2],s_d[2] = net.new_gene(0.369408,0,param,0)

   tf_bias = net.new_TFHill(s_d[1],3.215727,1.104619,tm_d[2]) #Switch -> Output
   
   param = [["TF",0],["Degradable",0.4574764],["Complexable"]]
   tm_d[3], prom_d[3], s_d[3] = net.new_gene(0.703658,0,param,0) # S3 sensor
   
   tf_bias2 = net.new_TFHill(s_d[2],3.266732,0.044938,tm_d[3])

   param =[["TF",0],["Degradable",4.244007]]
   PPI1, s_d[4] = net.new_PPI(s_d[2],s_d[3],6.783001,0.195168,param)

   param=[["TF",0],["Degradable",1.101876]]
   PPI2, s_d[5] = net.new_PPI(s_d[2],s_d[2],0.021174,1.075146,param)
  
   # Output 1 is the DIVISION decision maker
   #param = [["Species"],["Degradable",mutation.sample_dictionary_ranges("Species.degradation",net.Random)],["TF",1],["Output",1],["Complexable"]]
   #tm_d[3],prom_d[3],s_d[3] = net.new_custom_random_gene(param)
   #tf_bias = net.new_random_TFHill(s_d[1],tm_d[3])

   return net

def fitness_treatment(population):
    """Function to change the fitness of the networks"""
    pass
    #for nnetwork in range(population.npopulation):
    #    population.genus[nnetwork].fitness=eval(population.genus[nnetwork].data_evolution[0])
