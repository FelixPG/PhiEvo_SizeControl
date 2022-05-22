import pickle
import shelve
import glob,os,sys
import re
from phievo.AnalysisTools import main_functions as MF
from phievo.AnalysisTools  import palette
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from  matplotlib.lines import Line2D
import matplotlib.gridspec as gs
import numpy as np
import importlib.util
from phievo.Networks import mutation,classes_eds2
from phievo import  initialization_code
import phievo
from sklearn.linear_model import LinearRegression

class Simulation:
    """
    The simulation class is a container in which the informations about a simulation are unpacked. This is used for easy access to a simulation results.
    """
    def  __init__(self, path,mode="default"):
        """
        Creates a Simulation object. When th mode is not default (ex:test), the seeds are not loaded.
        usefull for prerun test.
        Args:
            path: directory of the project
            mode: Allows different mode to load the project.
        """
        self.root = path
        if self.root[-1] != os.sep:
            self.root += os.sep
        ## Upload Run parameters
        model_files = os.listdir(self.root)
        (model_dir , self.inits , init_file) =tuple(initialization_code.check_model_dir(self.root))
        self.inits.prmt["workplace_dir"] = os.path.join(self.inits.model_dir,"Workplace")
        phievo.Networks.mutation.dictionary_ranges.update(self.inits.dictionary_ranges)
        phievo.Networks.mutation.dictionary_ranges['CorePromoter.delay'] = int(
            0.5 + phievo.Networks.mutation.dictionary_ranges['CorePromoter.delay'] / self.inits.prmt['dt'])
        self.deriv2 = initialization_code.init_networks(self.inits)
        self.plotdata = initialization_code.import_module(self.inits.pfile['plotdata'])
        if mode in ["default"]:
            searchSeed  = re.compile("Seed(.+)$") ## integer ## at the end of the string "project_root/Seed##"
            seeds = [searchSeed.search(seed).group(1)  for seed in glob.glob(os.path.join(self.root,"Seed*"))]

            seeds.sort()
            seeds = [int(ss) for ss in seeds if ss.isdigit()] + [ss for ss in seeds if not ss.isdigit()]          
            if self.inits.prmt["pareto"]:
                self.type = "pareto"
                nbFunctions = self.inits.prmt["npareto_functions"]
                self.seeds = {seed:Seed_Pareto(os.path.join(self.root,"Seed{}".format(seed)),nbFunctions=nbFunctions) for seed in seeds}
            else:
                self.type = "default"
                self.seeds = {seed:Seed(os.path.join(self.root,"Seed{}".format(seed))) for seed in seeds}

        try:
            palette.update_default_colormap(self.inits.prmt["palette"]["colormap"])
        except KeyError:
            pass
        self.buffer_data = None



    def show_fitness(self,seed,smoothen=0,**kwargs):
        """Plot the fitness as a function of time

        Args:
            seed (int): the seed-number of the run

        Returns:
            matplotlib figure
        """
        fig = self.seeds[seed].show_fitness(smoothen,**kwargs)
        return fig

    def custom_plot(self,seed,X,Y):
        """Plot the Y as a function of X. X and Y can be chosen in the list ["fitness","generation","n_interactions","n_species"]

        Args:
            seed (int): number of the seed to look at
            X (str): x-axis observable
            Y (str): y-axis observable
        """
        x_val = self.seeds[seed].custom_plot(X,Y)

    def get_best_net(self,seed,generation):
        """ The functions returns the best network of the selected generation

        Args:
            seed (int): number of the seed to look at
            generation (int): number of the generation

        Returns:
            The best network for the selected generation
        """

        return self.seeds[seed].get_best_net(generation)

    def get_backup_net(self,seed,generation,index):
        """
            Get network from the backup file(or restart). In opposition to the best_net file
            the restart file is note stored at every generation but it contains a full
            population. This funciton allows to grab any individual of the population when
            the generation is stored

            Args:
                seed : index of the seed
                generation : index of the generation (must be a stored generation)
                index : index of the network within its generation
            Return:
                The selected network object
        """
        return self.seeds[seed].get_backup_net(generation,index)

    def get_backup_pop(self,seed,generation):
        """
        Cf get_backup_net. Get the complete population of networks for a generation that
        was backuped.

        Args:
            seed : index of the seed
            generation : index of the generation (must be a stored generation)
        Return:
            List of the networks present in the population at the selected generation
        """
        return self.seeds[seed].get_backup_pop(generation)

    def stored_generation_indexes(self,seed):
        """
        Return the list of the stored generation indexes

        Args:
            seed (int): Index of Seed, you want the stored generation for.
        Return:
            list of the stored generation indexes
        """
        return self.seeds[seed].stored_generation_indexes()

    def run_dynamics(self,net=None,trial=1,erase_buffer=False,return_treatment_fitness=False):
        """
        Run Dynamics for the selected network. The function either needs the network as an argument or the seed and generation information to select it. If a network is provided, seed and generation are ignored.

        Args:
            net (Networks): network to simulate
            trial (int): Number of independent simulation to run

        Returns: 
            data (dict) dictionnary containing the time steps
            at the "time" key, the network at "net" and the corresponding
            time series for index of the trial.
             - net : Network
             - time : time list
             - outputs: list of output indexes
             - inputs: list of input indexes
             - 0 : data for trial 0
                - 0 : array for cell 0:
                       g0 g1 g2 g3 ..
                    t0  .
                    t1     .
                    t2        .
                    .
                    .
        """
        if net is None:
            net = self.seeds[seed].get_best_net(generation)
        self.inits.prmt["ntries"] = trial
        prmt = dict(self.inits.prmt)
        N_cell = prmt["ncelltot"]
        N_species = len(net.dict_types['Species'])
        self.buffer_data = {"time":np.arange(0,prmt["dt"]*(prmt["nstep"]),prmt["dt"])}
        prmt["ntries"] = trial
        treatment_fitness = self.deriv2.compile_and_integrate(net,prmt,1000,True)
        col_select = np.arange(N_species)
        for i in range(trial):
            temp = np.genfromtxt('Buffer%d'%i, delimiter='\t')[:,1:]
            self.buffer_data[i] = {cell:temp[:,col_select + cell*N_species] for cell in range(N_cell)}
            if erase_buffer:
                os.remove("Buffer%d"%i)
            else:
                os.rename("Buffer{0}".format(i),os.path.join(self.root,"Buffer{0}".format(i)))

        self.buffer_data["net"] = net
        get_species = re.compile("s\[(\d+)\]")
        self.buffer_data["outputs"] = [int(get_species.search(species.id).group(1)) for species in net.dict_types["Output"]]
        self.buffer_data["inputs"] = [int(get_species.search(species.id).group(1)) for species in net.dict_types["Input"]]

        if return_treatment_fitness:
            return treatment_fitness
        return self.buffer_data


    def clear_buffer(self):
        """
        Clears the variable self.buffer_data.
        """
        self.buffer_data = None

    def PlotData(self,data,xlabel,ylabel,select_genes=[],no_popup=False,legend=True,lw=1,ax=None):
        """
        Function in charge of the call to matplotlib for both Plot_TimeCourse and Plot_Profile.
        """
        fig = plt.figure()
        if not ax:ax = fig.gca()
        Ngene = data.shape[1]
        colors = palette.color_generate(Ngene)
        for gene in range(Ngene):
            if select_genes!=[] and gene not in select_genes:
                continue
            ls = "--"
            label = "Species {}"
            if gene in self.buffer_data["outputs"]:
                ls = "-"
                label = "Output {}"
            if gene in self.buffer_data["inputs"]:
                ls = "-"
                label = "Input {}"
            ax.plot(data[:,gene],ls=ls,label=label.format(gene),lw=lw,color=colors[gene])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:ax.legend()
        return fig
    
    def Plot_TimeCourse(self,trial_index,cell=0,select_genes=[],no_popup=False,legend=True,lw=1,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and the cell and plot the gene time series

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            cell: Index of the cell to plot
            select_genes: list of gene indexes to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = self.buffer_data[trial_index]#[cell]
        try:
            data = data[cell]
        except KeyError:
            assert cell<0
            cell = sorted(list(data.keys()))[-1]
            data = data[cell]
        fig = self.PlotData(data,"Time","Concentration",select_genes=select_genes,no_popup=no_popup,legend=legend,lw=lw,ax=ax)        
        return fig

    def Plot_Profile(self,trial_index,time=0,select_genes=[],no_popup=False,legend=True,lw=1,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and plot the gene profile along the cells at
        the selected time point.

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            time: Index of the time to select
            select_genes: list of gene indexes to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = []
        for key,dynamics in sorted(self.buffer_data[trial_index].items()):
            data.append(dynamics[time,:])
        data = np.array(data)
        fig = self.PlotData(data,"Cell index","Concentration",select_genes=select_genes,no_popup=no_popup,legend=legend,lw=lw,ax=ax)
        return fig

        ################################
    ##############MODIF HERE################
        ################################

    
    def PlotScatter1(self,data,s=2.0,a=0.6,no_popup=False,legend=True,ax=None):
        """
        Function in charge of the call to matplotlib for Plot_MotherDaughter.
        """
        fig = plt.figure()
        if not ax:ax = fig.gca()
        volume = data[:,0]
        switch = data[:,1]
        if switch[0] == 1:
            ts = 1
        else:
            ts = 0
        vbirth = volume[0]
        sumx_birth = 0
        sumx2_birth = 0
        sumxy_g1 = 0
        sumxy_cycle = 0

        v_birth = [volume[0]]
        v_start = []
        v_div = []

        dummy_fitness = 0
        var = 0
        av = 0
        std=0
        dead = 0
        t_dead = data.shape[0]
        for t in range(data.shape[0]-1):
            if volume[t]<=0.1:
                dead = 0
                t_dead = t
                break
            if switch[t+1] != switch[t] and switch[t] == int(1-ts) and dead == 0:
                dummy_fitness += -1
                var+=volume[t]*volume[t]/4
                av+=volume[t]/2
        if dummy_fitness<0:
            var/=np.abs(dummy_fitness)
            av/=np.abs(dummy_fitness)
            std = np.sqrt(var - av**2)

        for t in range(len(volume)-1):
            if t <= t_dead:
            # Commit to START
                if (switch[t+1] - switch[t] == +1 and ts == 0) or (switch[t+1] - switch[t] == -1 and ts == 1):
                    vstart = volume[t+1]
                    sumxy_g1 += vbirth*(vstart-vbirth)
                    v_start.append(vstart)

            # Divide
                elif (switch[t+1] - switch[t] == -1 and ts == 0) or (switch[t+1] - switch[t] == 1 and ts == 1):
                    vdiv = volume[t]
                    sumxy_cycle += vbirth*(vdiv-vbirth)
                    vbirth = volume[t+1]
                    sumx2_birth += vbirth**2
                    v_div.append(vdiv)
                    v_birth.append(vbirth)
                    
        if len(v_birth)>len(v_start):
            # Divided without reaching START before reaching end of integration
            del v_birth[-1]
        elif len(v_start) > len(v_div):
            # Commited to START without a division before reaching end of integration
            del v_birth[-1], v_start[-1]

        #del size_diff_birth_start[0][0], size_diff_birth_start[1][0], size_diff_birth_div[0][0], size_diff_birth_div[1][0], size_diff_start_div[0][0], size_diff_start_div[1][0]
        added_vol_G1 = np.subtract(v_start,v_birth)
        added_vol_G2 = np.subtract(v_div,v_start)
        added_vol_cycle = np.subtract(v_div,v_birth)
        #print(v_birth,v_start,v_div)

        colors = ['#D01C1FFF','midnightblue','darkorchid']#palette.color_generate(3)
        linreg_G1 = LinearRegression().fit(np.array(v_birth[20:]).reshape(-1,1),np.array(added_vol_G1[20:]).reshape(-1,1))
        linreg_G2 = LinearRegression().fit(np.array(v_start[20:]).reshape(-1,1),np.array(added_vol_G2[20:]).reshape(-1,1))
        linreg_cycle = LinearRegression().fit(np.array(v_birth[20:]).reshape(-1,1),np.array(added_vol_cycle[20:]).reshape(-1,1))

        x_birth = np.linspace(np.min(v_birth[20:]),np.max(v_birth[20:]),10)
        x_start = np.linspace(np.min(v_start[20:]),np.max(v_start[20:]),10)
        
        y_G1 = linreg_G1.predict(x_birth.reshape(-1,1))
        y_G2 = linreg_G2.predict(x_start.reshape(-1,1))
        y_cycle = linreg_cycle.predict(x_birth.reshape(-1,1))
        
        ax.scatter(v_birth[20:],added_vol_G1[20:],s=s,alpha=a,label='G1, slope = %.2f'%(linreg_G1.coef_[0]),color=colors[0])
        ax.scatter(v_start[20:],added_vol_G2[20:],s=s,alpha=a,label='S/G2/M, slope = %.2f'%(linreg_G2.coef_[0]),color=colors[1])
        ax.scatter(v_birth[20:],added_vol_cycle[20:],s=s,alpha=a,label='Cell cycle, slope = %.2f'%(linreg_cycle.coef_[0]),color=colors[2])
        ax.plot(x_birth,y_G1,color=colors[0],linewidth=2,alpha=a)
        ax.plot(x_start,y_G2,color=colors[1],linewidth=2,alpha=a)
        ax.plot(x_birth,y_cycle,color=colors[2],linewidth=2,alpha=a)

        ax.set_xlabel("Initial Volume")
        ax.set_ylabel("Added Volume")
        ax.set_title("Added volume during different phases of the cell cycle, dead at t = %f, n_div = %i"%(t_dead*0.05,int(dummy_fitness)))
        if legend:ax.legend()
        return fig

    def Plot_SizeDiff(self,trial_index,cell=0,no_popup=False,legend=True,s=2.0,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and the cell and plot the gene time series

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            cell: Index of the cell to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = self.buffer_data[trial_index]#[cell]
        try:
            data = data[cell]
        except KeyError:
            assert cell<0
            cell = sorted(list(data.keys()))[-1]
            data = data[cell]
        fig = self.PlotScatter1(data,s=s,no_popup=no_popup,legend=legend,ax=ax)        
        return fig

    def PlotScatter2(self,data,s=2.0,a=0.6,no_popup=False,ax=None):
        """
        Function in charge of the call to matplotlib for Plot_Whi5SizeCorrelations.
        """
        fig = plt.figure()
        GS = gs.GridSpec(3,2,wspace=0.20,hspace=0.3,figure=fig)
        ax1 = fig.add_subplot(GS[0,0])
        ax2 = fig.add_subplot(GS[0,1])
        ax3 = fig.add_subplot(GS[1,0])
        ax4 = fig.add_subplot(GS[1,1])
        ax5 = fig.add_subplot(GS[2,0])
        ax6 = fig.add_subplot(GS[2,1])
        #if not ax:ax = fig.gca()
        volume = data[:,0]
        switch = data[:,1]
        whi5 = data[:,2]
        s3 = data[:,3]

        sizebirth_Whi5 = [[volume[0]],[whi5[0]]]
        Whi5Qty_SG2M = []
        sizestart_Whi5 = [[],[]]
        Whi5Qty_G1 = []
        Whi5Qty_cycle = []
        g1_lengths = []
        cycle_lengths = []
        sizebirth_s3 = [[volume[0]],[s3[0]]]
        sizestart_s3 = [[],[]]
        t0 = 0
        dt = 0.05
        Whi5_init = whi5[0]*volume[0]
        dead = 0
        t_dead = data.shape[0]
        for t in range(data.shape[0]-1):
            # Commit to START
            if volume[t]<=0.5:
                dead = 1
                t_dead = t
                break
            elif switch[t+1] != switch[t] and switch[t] == 0:
                sizestart_Whi5[0].append(volume[t+1])
                sizestart_Whi5[1].append(whi5[t+1])
                sizestart_s3[0].append(volume[t+1])
                sizestart_s3[1].append(s3[t+1])
                Whi5Qty_G1.append(whi5[t+1]*volume[t+1]-Whi5_init)
                g1_lengths.append((t-t0+1)*dt)
            # Divide
            elif switch[t+1] != switch[t] and switch[t] == 1:
                sizebirth_Whi5[0].append(volume[t]/2)
                sizebirth_Whi5[1].append(whi5[t])
                sizebirth_s3[0].append(volume[t]/2)
                sizebirth_s3[1].append(s3[t])
                Whi5Qty_SG2M.append(whi5[t]*volume[t]-Whi5Qty_G1[-1]+Whi5_init)
                Whi5Qty_cycle.append(whi5[t]*volume[t]-Whi5_init)
                cycle_lengths.append((t-t0)*dt)
                t0 = t+1
                Whi5_init = whi5[t]*volume[t]/2
        if len(g1_lengths)<len(sizebirth_Whi5[0]):
            # Divided without reaching START before reaching end of integration
            del sizebirth_Whi5[0][-1],sizebirth_Whi5[1][-1], sizebirth_s3[0][-1],sizebirth_s3[1][-1]
        if len(sizebirth_Whi5[0])>len(Whi5Qty_SG2M):
            # Reached START before division before reaching end of integration
            del sizebirth_Whi5[0][-1],sizebirth_Whi5[1][-1],g1_lengths[-1],sizestart_Whi5[0][-1],sizestart_Whi5[1][-1], Whi5Qty_G1[-1], sizebirth_s3[0][-1],sizebirth_s3[1][-1], sizestart_s3[0][-1],sizestart_s3[1][-1]

        colors = palette.color_generate(6)
        x_sizebirth = np.linspace(np.min(sizebirth_Whi5[0]),np.max(sizebirth_Whi5[0]),50)
        g1_adder_length = np.log(np.divide(np.mean(np.subtract(sizestart_Whi5[0],sizebirth_Whi5[0]))*np.ones(50),x_sizebirth)+np.ones(50))/0.25

        ax1.scatter(sizebirth_Whi5[0][1:],sizebirth_Whi5[1][1:],s=s,alpha=a,color=colors[0])
        ax2.scatter(sizebirth_Whi5[0][1:],np.multiply(sizebirth_Whi5[1][1:],sizebirth_Whi5[0][1:]),s=s,alpha=a,color=colors[1])
        ax3.scatter(sizestart_Whi5[0][1:],sizestart_Whi5[1][1:],s=s,alpha=a,color=colors[2])
        ax4.scatter(sizebirth_Whi5[0][1:],g1_lengths[1:],s=s,alpha=a,color=colors[3])
        #ax4.plot(x_sizebirth,g1_adder_length,alpha=a,color=colors[3])
        ax5.scatter(sizestart_s3[0][1:],sizestart_s3[1][1:],s=s,alpha=a,color=colors[4])
        ax6.scatter(sizestart_s3[0][1:],np.multiply(sizestart_s3[1][1:],sizestart_s3[0][1:]),s=s,alpha=a,color=colors[5])

        ax1.set_xlabel("Volume at birth")
        ax2.set_xlabel("Volume at birth")
        ax3.set_xlabel("Volume at START")
        ax4.set_xlabel("Volume at birth")
        ax5.set_xlabel("Volume at START")
        ax6.set_xlabel("Volume at START")
        
        ax1.set_ylabel(r"$[Output]_{birth}$")
        ax2.set_ylabel(r"$Output_{birth}$")
        ax3.set_ylabel(r"$[Output_{START}]$")
        ax4.set_ylabel("G1 length (hr)")
        ax5.set_ylabel(r"$[S_3]_{START}$")
        ax6.set_ylabel(r"$S_3_{START}$")
        #ax3.set_ylim(0,0.25)

        #ax1.set_xlim(0.5,np.max(sizebirth_Whi5[0])+0.5),ax2.set_xlim(0.5,np.max(sizebirth_Whi5[0])+0.5),ax3.set_xlim(0.5,np.max(sizestart_Whi5[0])+0.5),ax4.set_xlim(0.5,np.max(sizebirth_Whi5[0])+0.5)
        #ax5.set_xlim(0.5,np.max(sizebirth_Whi5[0])+0.5), ax6.set_xlim(0.5,np.max(sizestart_Whi5[0])+0.5)
        #ax1.set_ylim(0.55,3)#,ax2.set_ylim(0.95,3)#,ax4.set_ylim(1,5)
        return fig

    def Plot_Whi5SizeCorrelations(self,trial_index,cell=0,no_popup=False,s=2.0,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and the cell and plot the gene time series

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            cell: Index of the cell to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = self.buffer_data[trial_index]#[cell]
        try:
            data = data[cell]
        except KeyError:
            assert cell<0
            cell = sorted(list(data.keys()))[-1]
            data = data[cell]
        fig = self.PlotScatter2(data,s=s,no_popup=no_popup,ax=ax)        
        return fig

    def PlotDistributions(self,data,a=0.6,no_popup=False,ax=None):
        """
        Function in charge of the call to matplotlib for Plot_Whi5SizeCorrelations.
        """
        fig = plt.figure()
        GS = gs.GridSpec(3,1,wspace=0.20,hspace=0.3,figure=fig)
        ax1 = fig.add_subplot(GS[0])
        ax2 = fig.add_subplot(GS[1])
        ax3 = fig.add_subplot(GS[2])

        #if not ax:ax = fig.gca()
        volume = data[:,0]
        switch = data[:,1]

        sizebirth = [volume[0]]
        sizestart = []
        sizediv = []

        t0 = 0
        dt = 0.05

        dead = 0
        t_dead = data.shape[0]

        if switch[0] == 1:
            ts = 1
        else:
            ts = 0

        for t in range(data.shape[0]-1):
            # Commit to START
            if volume[t]<=0.5:
                dead = 1
                t_dead = t
                break
            elif ((switch[t+1]-switch[t]) == 1 and not ts) or ((switch[t+1]-switch[t]) == -1 and  ts):
                sizestart.append(volume[t])
            # Divide
            elif ((switch[t+1]-switch[t]) == -1 and not ts) or ((switch[t+1]-switch[t]) == 1 and  ts):
                sizediv.append(volume[t])
                sizebirth.append(volume[t]/2)

        av_sizebirth = np.mean(sizebirth)
        av_sizestart = np.mean(sizestart)
        av_sizediv = np.mean(sizediv)

        std_sizebirth = np.std(sizebirth)
        std_sizestart = np.std(sizestart)
        std_sizediv = np.std(sizediv)

        if av_sizebirth != 0:
            cv_sizebirth = std_sizebirth/av_sizebirth
        else:
            cv_sizebirth = 0
        if av_sizestart != 0:
            cv_sizestart = std_sizestart/av_sizestart
        else:
            cv_sizestart = 0
        if av_sizediv != 0:
            cv_sizediv = std_sizediv/av_sizediv
        else:
            cv_sizediv = 0

        colors = palette.color_generate(3)
        xlim1,xlim2 = np.min(sizebirth[1:]),np.max(sizediv[1:])
        bins = np.linspace(xlim1,xlim2,40)

        counts1, bins1, f1 = ax1.hist(sizebirth[1:],bins,alpha=a,color=colors[0])
        counts2, bins2, f2 = ax2.hist(sizestart[1:],bins,alpha=a,color=colors[1])
        counts3, bins3, f3 = ax3.hist(sizediv[1:],bins,alpha=a,color=colors[2])
        l1, l2, l3 = np.max(counts1)/2,np.max(counts2)/2,np.max(counts3)/2
        
        lab1 = 'Av = %.2f, Std = %.2f, CV = %.3f'%(av_sizebirth,std_sizebirth,cv_sizebirth)
        lab2 = 'Av = %.2f, Std = %.2f, CV = %.3f'%(av_sizestart,std_sizestart,cv_sizestart)
        lab3 = 'Av = %.2f, Std = %.2f, CV = %.3f'%(av_sizediv,std_sizediv,cv_sizediv)
        ax1.axvline(av_sizebirth,color='r')
        ax2.axvline(av_sizestart,color='r')
        ax3.axvline(av_sizediv,color='r')
        ax1.hlines(y=l1,xmin=(av_sizebirth-cv_sizebirth),xmax=(av_sizebirth+cv_sizebirth),linestyle='--',color='r',label=lab1)
        ax2.hlines(y=l2,xmin=(av_sizestart-cv_sizestart),xmax=(av_sizestart+cv_sizestart),linestyle='--',color='r',label=lab2)
        ax3.hlines(y=l3,xmin=(av_sizediv-cv_sizediv),xmax=(av_sizediv+cv_sizediv),linestyle='--',color='r',label=lab3)
        ax1.set_xlabel("Volume at birth")
        ax2.set_xlabel("Volume at START")
        ax3.set_xlabel("Volume at division")
        ax1.set_ylabel("Counts")
        ax2.set_ylabel("Counts")
        ax3.set_ylabel("Counts")
        ax1.set_xlim([xlim1,xlim2])
        ax2.set_xlim([xlim1,xlim2])
        ax3.set_xlim([xlim1,xlim2])
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.suptitle('Size distributions at birth, START and division')

        return fig

    def Plot_SizeDistributions(self,trial_index,cell=0,no_popup=False,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and the cell and plot the gene time series

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            cell: Index of the cell to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = self.buffer_data[trial_index]#[cell]
        try:
            data = data[cell]
        except KeyError:
            assert cell<0
            cell = sorted(list(data.keys()))[-1]
            data = data[cell]
        fig = self.PlotDistributions(data,no_popup=no_popup,ax=ax)        
        return fig

    def PlotQty(self,data,xlabel,ylabel,select_genes=[],no_popup=False,legend=True,lw=1,ax=None):
        """
        Function in charge of the call to matplotlib for both Plot_TimeCourseQty.
        """
        fig = plt.figure()
        if not ax:ax = fig.gca()
        Ngene = data.shape[1]
        colors = palette.color_generate(Ngene)
        for gene in range(Ngene):
            if select_genes!=[] and gene not in select_genes:
                continue
            ls = "--"
            label = "Species {}"
            if gene in self.buffer_data["outputs"]:
                ls = "-"
                label = "Output {}"
            if gene in self.buffer_data["inputs"]:
                ls = "-"
                label = "Input {}"
            if gene == 0:
                ax.plot(data[:,gene],ls=ls,label=label.format(gene),lw=lw,color=colors[gene])
            else: 
                ax.plot(np.multiply(data[:,gene],data[:,0]),ls=ls,label=label.format(gene),lw=lw,color=colors[gene])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:ax.legend()
        return fig
    
    def Plot_TimeCourseQty(self,trial_index,cell=0,select_genes=[],no_popup=False,legend=True,lw=1,ax=None):
        """
        Searches in the data last stored in the Simulation buffer for the time course
        corresponding to the trial_index and the cell and plot the gene time series

        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            cell: Index of the cell to plot
            select_genes: list of gene indexes to plot
            no_popup: False by default. Option used to forbid matplotlib popup windows
                     Useful when saving figures to a file.
        Return:
            figure
        """
        data = self.buffer_data[trial_index]#[cell]
        try:
            data = data[cell]
        except KeyError:
            assert cell<0
            cell = sorted(list(data.keys()))[-1]
            data = data[cell]
        fig = self.PlotQty(data,"Time","Quantity",select_genes=select_genes,no_popup=no_popup,legend=legend,lw=lw,ax=ax)        
        return fig

        ################################
    ##############MODIF HERE################
        ################################

    def load_Profile_data(self,trial_index,time):
        """
        Loads the data from the simulation and generate ready to plot data.
        Args:
            trial_index: index of the trial you. Refere to run_dynamics to know how
            many trials there are.
            time: Index of the time to select
        """
        data = []
        for key,dynamics in sorted(self.buffer_data[trial_index].items()):
            data.append(dynamics[time,:])
        return np.array(data)
        
    def get_genealogy(self,seed):
        return Genealogy(self.seeds[seed])

class Seed:
    """
    This is a container to load the information about a Simulation seed. It contains mainly the indexes of the generations and some extra utilities to analyse them.
    """


    def  __init__(self, path):
        self.root = path
        if self.root[-1] != os.sep:
            self.root += os.sep

        self.restart_path = self.root + "Restart_file"
        self.name = re.search("[^/.]*?(?=/?$)",path).group(0)

        data = shelve.open(self.root+"data")
        indexes = data["generation"]
        interac = data['n_interactions']
        species = data['n_species']
        fitness = data['fitness']

        self.generations = {
            i:{
                "n_interactions" : interac[i],
                "n_species" : species[i],
                "fitness" : fitness[i]
                }
            for i in indexes}
        self.indexes = indexes
        
        self.observables = {
            "generation":lambda:list(self.indexes),
            "n_interactions":lambda:[gen["n_interactions"] for i,gen in self.generations.items()],
            "n_species":lambda:[gen["n_species"] for i,gen in self.generations.items()],
            "fitness":lambda:[gen["fitness"] for i,gen in self.generations.items()],
        }
        self.default_observable = "fitness"

        with shelve.open(self.restart_path) as data:
            self.pop_size = len(data["0"][1])
            self.restart_generations = sorted([int(xx) for xx in data.dict.keys()])

    def show_fitness(self,smoothen=0,**kwargs):
        """Plot the fitness as a function of time
        """
        self.custom_plot("generation","fitness")



    def custom_plot(self,X,Y):
        """Plot the Y as a function of X. X and Y can be chosen in the keys of
            self.observables.

        Args:
            seed (int): number of the seed to look at
            X (str): x-axis observable
            Y (list): list (or string) of y-axis observable
        """
        x_val = self.observables[X]()
        if isinstance(Y,str):
            Y = [Y]
        Y_val = {y:self.observables[y]() for y in Y}

        NUM_COLORS = len(Y)
        color_l= {Y[i]:col for i,col in zip(range(NUM_COLORS),palette.color_generate(NUM_COLORS))}
        fig = plt.figure()
        ax = fig.gca()
        for label,y_val in Y_val.items():
            if y_val is not None:
                ax.plot(x_val,y_val,lw=2,color=color_l[label],label=label)
        ax.set_xlabel(X)
        ax.set_ylabel(Y[0])
        ax.legend()
        fig.show()
        return fig

    def get_best_net(self,generation):
        """ The functions returns the best network of the selected generation

        Args:
            seed (int): number of the seed to look at

        Returns:
            Networks : the best network for the selected generation
        """

        return MF.read_network(self.root+"Bests_%d.net"%generation,verbose=False)

    def get_backup_net(self,generation,index):
        """
        Get network from the backup file(or restart). In opposition to the best_net file
        the restart file is note stored at every generation but it contains a full
        population. This funciton allows to grab any individual of the population when
        the generation is stored

        Args:
            generation : index of the generation (must be a stored generation)
            index : index of the network within its generation
        Return:
            the selected network object
        """
        with shelve.open(self.restart_path) as data:
            dummy,nets = data[str(generation)]
        return(nets[index])

    def get_backup_pop(self,generation):
        """
        Cf get_backup_net. Get the complete population of networks for a generation that
        was backuped.

        Args:
            generation : index of the generation (must be a stored generation)
        Return:
            List of the networks present in the population at the selected generation
        """
        with shelve.open(self.restart_path) as data:
            dummy,nets = data[str(generation)]
        return nets

    def stored_generation_indexes(self):
        """
        Return the list of the stored generation indexes

        Return:
            list of the stored generation indexes
        """
        return list(self.restart_generations)

    def compute_best_fitness(self,generation):
        pass


class Seed_Pareto(Seed):
    def __init__(self,path,nbFunctions):
        super(Seed_Pareto, self).__init__( path)
        self.nbFunctions = nbFunctions

        self.observables.pop("fitness")

        for ff in range(self.nbFunctions):
            self.observables["fitness{0}".format(ff)] = lambda ff=ff:[gen["fitness"][ff] for i,gen in self.generations.items()]
        self.default_observable = "fitness1"

    def show_fitness(self,smoothen=0,index=None):
        """Plot the fitness as a function of time

        Args:
            seed (int): the seed-number of the run
            index(array): index of of the fitness to plot. If None, all the fitnesses are ploted
        Returns:
            Matplolib figure
        """
        gen = self.get_observable("generation")

        fit = np.array(self.get_observable("fitness"))

        if not index :
            index = range(fit.shape[1])

        fig = plt.figure()
        ax = fig.gca()

        for i in index:
            if smoothen:
                fit = MF.smoothing(fit,smoothen)

            ax.plot(gen,fit[:,i],lw=2,label='fitness%d'%i)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend(loc='upper right')
        fig.show()
        return fig

    def pareto_generate_fit_dict(self,generations,max_rank=1):
        """
        Load fitness data for the selected generations and format them to be
        understandable by plot_pareto_fronts
        """
        data = MF.load_generation_data(generations,self.root+"Restart_file")        
        fitnesses = {gen:
                     [
                         [net.fitness
                          for ind,net in enumerate(networks) if net.prank==rank+1]
                      for rank in range(min(max_rank,3))]                      
                     for gen,networks in data.items()}
        net_info  = {gen:
                     [
                         [dict(gen=gen,net=ind,rank=net.prank,F1=net.fitness[0],F2=net.fitness[1],ind=net.identifier)
                          for ind,net in enumerate(networks) if net.prank==rank+1]
                      for rank in range(min(max_rank,3))]                      
                     for gen,networks in data.items()}
        return net_info,fitnesses
    
    def plot_pareto_fronts(self,generations,max_rank=1,with_indexes=False,legend=False,xlim=[],ylim=[],colors=[],gradient=[],xlabel="F_1",ylabel="F_2",s=50,no_popup=False):
        """
        Plot every the network of the selected generations in the (F_1,F_2) fitness space.

        Args:
            generations (list): list of the selected generations
            max_rank (int): In given population plot only the network of rank <=max_rank
            with_indexes(bool): NotImplemented 
            legend(bool): NotImplemented
            xlim (list): [xmax,xmin]
            ylim (list): [ymax,ymin]
            colors (list): List of html colors, one for each generation
            gradient (list): List of colors to include in the gradient
            xlabel(str): Label of the xaxis
            ylabel(str): Label of the yaxis
            s (float): marker size
            no_popup(bool): prevents the popup of the plot windows 
        
        Returns:
            matplotlib figure
        """
        
        net_info,fitnesses = self.pareto_generate_fit_dict(generations,max_rank)
        if not colors and not gradient:colors = {gen:col for gen,col in zip(fitnesses.keys(),palette.color_generate(len(fitnesses)))}
        if gradient:
            colors = {gen:col for gen,col in zip(generations,palette.generate_gradient(generations,gradient))}
            
        shapes = ["o","s","^"]
        fig = plt.figure()
        ax = fig.gca()
        for gen,ranks in sorted(fitnesses.items(), key=lambda x:x[1]):
            for ind,rank in enumerate(ranks):
                if not rank: continue
                ######## MODIF HERE, BEFORE ONLY F1, F2##########
                if len(rank) == 3:
                    F1,F2,F3 = zip(*rank)
                    ax.scatter(F2,F3,s=s,color=colors[gen],marker=shapes[ind])
                else:
                    F1,F2 = zip(*rank)
                    ax.scatter(F1,F2,s=s,color=colors[gen],marker=shapes[ind]) 
                    ######## MODIF HERE, BEFORE ONLY F1, F2##########               
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:ax.set_xlim(xlim)
        if ylim:ax.set_ylim(ylim)
        if not no_popup: plt.show()
        return fig
        
        

class Genealogy:
    def __init__(self,seed):
        self.generations = seed.stored_generation_indexes()
        assert self.generations == list(range(min(self.generations),max(self.generations)+1)),"\n\tA Genealogy object cannot be created from a seed where not all the generations were stored.\n\tMake sure prmt['restart']['freq'] = 1 in the init file."
        self.root = seed.root
        self.restart_path = seed.restart_path
        self.seed = seed
        self.networks = {}

    def sort_networks(self,verbose=False,write_pickle=True):
        """
        Order the networks, by the label_ind, in a dictionary.
        The dictonary contains the most useful information but takes last space.
        The information dictionaries is easier to handle than the actual networks.

        Args:
            verbose: print information during sorting
            write_pickle: backup the sorting information in a pickle file
        Return:
            dictionary. A key is associated to each network
        """
        networks = {}
        with shelve.open(self.restart_path) as data:            
            for gen in self.generations:
                dummy,population = data[str(gen)]
                
                for i,net in enumerate(population):
                    net_id = net.identifier
                    try:
                        networks[net_id]
                    except KeyError:                        
                        networks[net_id] = dict(
                            ind = net_id,
                            gen = gen,
                            par = net.parent,
                            fit = net.fitness,
                            pos = i,
                            las = net.last_mutation if net.last_mutation else [] 
                        )
                if verbose and (gen%100==0):
                    print("Generation\t{}/{} done.".format(gen,self.generations[-1]))
                    
        if write_pickle:
            with open(os.path.join(self.root,"networks_info.pkl"),"wb") as pkl_networks:
                pickle.dump(networks,pkl_networks)
        self.networks = networks
        return networks

    def load_sort_networks(self):
        """Loads an existing network classification"""
        with open(os.path.join(self.root,"networks_info.pkl"),"rb") as pkl_networks:
            networks = pickle.load(pkl_networks)
        self.networks = networks
        return networks

    def search_ancestors(self,network):
        if type(network) is int:
            network = self.networks[network]
        ancestors = [network]        
        while True:    
            try:
                network = self.networks[network["par"]]
                ancestors = [network] + ancestors
            except KeyError:
                break
        return ancestors
    
    
    def plot_front_genealogy(self,generations,extra_networks_info=[],filename=""):
        """
        Uses the seed plot_pareto_fronts function to display the pareto fronts.
        In addition, the function allows to plots extra networks in the fitness plan
        
        Args:
            generations: list of generation indexes
            extra_networks_indexes: list of extra network informatino dictionaries.

        """
        from phievo.AnalysisTools import plotly_graph
        fig = self.seed.plot_pareto_fronts(generations,with_indexes=True)
        if extra_networks_info:
            fit0 = [net_inf["fit"][0] for net_inf in extra_networks_info]
            fit1 = [net_inf["fit"][1] for net_inf in extra_networks_info]

            trace = plotly_graph.go.Scatter(x=fit0,y=fit1,mode = 'markers',name="Extra networks",
                                            marker= dict(size=9,color= "black",symbol="square"),
                                            hoverinfo="text",
                                            legendgroup = "Extra networks",
                                            text=["net #{}\nmutation:{}".format(net_inf["ind"]," - ".join(net_inf["las"] if net_inf["las"] else [])) for net_inf in extra_networks_info]
            )
            fig.data.append(trace)
            if filename:
                plotly_graph.py.plot(fig,filename=filename)
            else:
                plotly_graph.py.plot(fig)

    def plot_mutation_fitness_deviation(self,only_one_mutation=True,networks=None,ploted_ratio=1):
        """
        Plot the deviation of fitness in the fitness space caused by a generation's mutation.
        
        Arg:
           only_one_mutation (bool): If True, plot only the networks that undergone only a single mutation durign a generation.

        """
        from phievo.AnalysisTools import plotly_graph
        dict_data = {}
        if networks:
            if type(networks)==list:
                networks = {net["ind"]:net  for net in networks}
        else:
            networks = self.networks
            
        for net_ind,net_inf in networks.items():
            if net_inf["las"] is None or (only_one_mutation and len(net_inf["las"])!=1):
                continue
            label = "-".join(net_inf["las"])
            try:
                parent = networks[net_inf["par"]]
            except KeyError:
                ## Parent not provided in the dict
                continue
            diff = [net_inf["fit"][0]-parent["fit"][0],net_inf["fit"][1]-parent["fit"][1]]
            data =  {"diff":diff,"label":"Net #{}\nparent #{}\nmutation: {}\nfitness change:{}".format(net_inf["ind"],parent["ind"],label,str(diff)) }
            dict_data[label] = dict_data.get(label,[]) + [data]

        plot_list = []
        colors = palette.color_generate(len(dict_data))

        for mut_ind,mut_name in enumerate(dict_data.keys()):
            mutation = dict_data[mut_name]
            x_val = [mut["diff"][0] for mut in mutation]
            y_val = [mut["diff"][1] for mut in mutation]
            hover_info = [mut["label"] for mut in mutation]
            L = len(x_val)
            if ploted_ratio!=1:
                selected_indexes = np.random.choice(range(L),int(L*ploted_ratio),replace=False)
                x_val = np.array(x_val)[selected_indexes]
                y_val = np.array(y_val)[selected_indexes]
                hover_info = np.array(hover_info)[selected_indexes]
            trace = plotly_graph.go.Scatter(x = x_val,y=y_val,mode = 'markers',name=mutation,
                                            marker= dict(size=14,color=colors[mut_ind]),
                                            hoverinfo="text",
                                            text=hover_info,
            )
            plot_list.append(trace)
            
        plotly_graph.py.plot(plot_list)

    def get_network_from_identifier(self,net_ind):
        try:
            network = self.networks[net_ind]
        except KeyError:
            raise KeyError("Index {} corresponds to no stored network.".format(net_ind))
        net = self.seed.get_backup_net(network["gen"],network["pos"])
        assert net.identifier==net_ind,"\tBug in get_network_from_identifier.\n\tThe obtained network has indec {} instead of {}.".format(net.identifier,net_ind)
        return net
    
    def plot_lineage_fitness(self,line,formula="{}",highlighted_mutations = []):
        from phievo.AnalysisTools import plotly_graph
        generations = [net_inf["gen"] for net_inf in line]
        fitness = [eval(formula.format(net_inf["fit"])) for net_inf in line]
        generate_hoverinfo = lambda net:"net #{}\nmutations:{}".format(net["ind"],"-".join(net["las"] if net["las"] else [] ))
        hover_info = [generate_hoverinfo(net_inf)  for net_inf in line]
        colors = palette.color_generate(len(highlighted_mutations)+1)
        trace = plotly_graph.go.Scatter(
            x = generations,
            y = fitness,
            text = hover_info,
            mode = "markers+lines",
            marker = dict(color=colors[0]),
        )
        data_to_plot = [trace]
        if highlighted_mutations:
            dict_highlighted = {mut:plotly_graph.go.Scatter(
                x = [],
                y = [],
                text = [],
                mode = "markers",
                marker = dict(color=colors[i+1]),
            ) for i,mut in enumerate(highlighted_mutations)}

            for net_inf in line:
                try:
                    ind = "-".join(net_inf["las"]) if net_inf["las"] else ""
                    trace = dict_highlighted[ind]
                except KeyError:
                    continue
                trace["x"].append(net_inf["gen"])
                trace["y"].append(eval(formula.format(net_inf["fit"])))
                trace["text"].append(generate_hoverinfo(net_inf))
            data_to_plot += list(dict_highlighted.values())
        plotly_graph.py.plot(data_to_plot)
        
    def plot_compare_multiple_networks(self,sim,indexes,cell=0):
        """
        Print a svg figure of the cell profile,time series and the network layout in
        the seed folder.
        """
        
        fig_format = "svg"
        for filename in glob.glob(os.path.join(self.root,"*svg")):
            os.remove(filename)
        fig_name = lambda xx,ind: os.path.join(self.root,"{}{}.{}".format(xx,ind,fig_format))
        for i,net_ind in enumerate(indexes):
            net = self.get_network_from_identifier(net_ind)
            net.draw(fig_name("net",net.identifier))
            res = sim.run_dynamics(net)
            fig_profile=sim.Plot_Profile(0,time=2999,no_popup=True)
            fig_profile.savefig(fig_name("profile",net.identifier))
            fig_time_course=sim.Plot_TimeCourse(0,cell=cell,no_popup=True)
            fig_time_course.savefig(fig_name("timecourse",net.identifier))        
            del fig_profile,fig_time_course

    def compare_ss_wrt_parent(self,sim,child,parent):
        from phievo.AnalysisTools import plotly_graph
        child =  self.get_network_from_identifier(child)
        parent = self.get_network_from_identifier(parent)
        res = sim.run_dynamics(child)
        child_profile = sim.Plot_Profile(0,time=2999,no_popup=True)
        res = sim.run_dynamics(parent)
        parent_profile = sim.Plot_Profile(0,time=2999,no_popup=True)
        for dat_par,dat_chi in zip(child_profile["data"],parent_profile["data"]):
            chi_y = np.array(dat_chi["y"])
            par_y = np.array(dat_par["y"])
            relative_y =  np.array(chi_y - par_y)/np.where(par_y==np.array(0),1,par_y)
            dat_chi["y"] = relative_y
        plotly_graph.py.plot(parent_profile)
        
    def scatter_pareto_accross_generations(self,generation,front_to_plot,xrange,yrange,step=1):
        from phievo.AnalysisTools import plotly_graph
        plotly_graph.py.init_notebook_mode(connected=True)
        generation_identifiers = [net.identifier for net in self.seed.get_backup_pop(generation)]
        generation_info = [self.networks[net_ind] for net_ind in generation_identifiers]
        
        data_to_plot = []
        steps = []
        full_scatter =dict(
            name = generation,
            mode = "markers",
            x = [],
            y = [],
            marker = dict(color="black",size=2),
        )
        for gen in front_to_plot:
            generation_identifiers = [net.identifier for net in self.seed.get_backup_pop(gen)]
            networks = [self.networks[net_ind] for net_ind in generation_identifiers]
            fit0 = [net["fit"][0] for net in networks]
            fit1 = [net["fit"][1] for net in networks]
            full_scatter["x"]+=fit0
            full_scatter["y"]+=fit1
            
        while True:
            for net_pos in range(len(generation_info)):
                net_inf = generation_info[net_pos]
                
                while net_inf["gen"] > generation:
                    generation_info[net_pos] = self.networks[net_inf["par"]]
                    assert generation_info[net_pos]["ind"] == net_inf["par"]                    
                    assert generation_info[net_pos], "\n\tscatter_pareto_accross_generations found no parent for network #{}(generation {}).".format(net_inf["ind"],generation)
                    net_inf = self.networks[net_inf["par"]]
                    
            fit0 = [net_inf["fit"][0] for net_inf in generation_info]
            fit1 = [net_inf["fit"][1] for net_inf in generation_info]
            networks_info = ["#{} parent:#{} fitness:{}".format(net_inf["ind"],net_inf["par"],net_inf["fit"].__str__()) for net_inf in generation_info]
            generation_dict =dict(
                name = generation,
                mode = "markers",
                x = fit0,
                y = fit1,
                text=networks_info
            )

            slider_step = {'args': [
                [generation],
                {'frame': {'duration': 300, 'redraw': False},
                 'mode': 'immediate',
                 'transition': {'duration': 300}}
            ],
                           'label': generation,
                           'method': 'animate'}
            data_to_plot.append(generation_dict)
            steps.append(slider_step)
            
            
            generation -= step
            if generation < 0:
                break
        figure = {
            'data': [full_scatter,data_to_plot[0]],
            'layout': {
                'xaxis' : {'range': xrange, 'title': 'Fitness 1'},
                'yaxis' : {'range': yrange, 'title': 'Fitness 2'},
                'hovermode':'closest',
                'updatemenus':[
                    {
                        'buttons': [
                            {
                                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                                'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                                'label': 'Play',
                                'method': 'animate'
                            },
                            {
                                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                                  'transition': {'duration': 0}}],
                                'label': 'Pause',
                                'method': 'animate'
                            }
                        ],
                        'direction': 'left',
                        'pad': {'r': 10, 't': 87},
                        'showactive': False,
                        'type': 'buttons',
                        'x': 0.1,
                        'xanchor': 'right',
                        'y': 0,
                        'yanchor': 'top'
                    } 
                ],
                'sliders':[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 20},
                        'prefix': 'Generation:',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': steps[::-1],
                }]
            },
            'frames': [{"data":[full_scatter,dat],"name":dat["name"]}
                for dat in data_to_plot[::-1]
            ],
        }
        
        
        plotly_graph.py.plot(figure, filename='pareto_accross_generations.html')
        return figure
        
