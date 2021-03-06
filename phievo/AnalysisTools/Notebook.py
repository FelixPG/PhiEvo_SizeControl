import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from  ipywidgets import widgets
from ipywidgets import interact, interactive, fixed
from IPython.display import display,HTML,clear_output
import os
from phievo.AnalysisTools import Simulation
## Load plotly_graph in order to use plotly in the notebook
import phievo.AnalysisTools.plotly_graph as plotly_graph
import phievo
import re
plotly_graph.run_in_nb()


found_str = "<p style=\"color:#31B404;font-size: 30px;\">✔</p>"
notfound_str = "<p style=\"color:#DF3A01;font-size: 30px;\">✘</p>"

class Notebook(object):
    """
        Wrapper that contains both the the widgets and  the simulation results.
        This way it is easy to update the state of the widgets when you load a
        new simulation
    """
    def __init__(self):
        self.sim = None
        rcParams['figure.figsize'] = (9.0, 8.0)
        self.project = None
        self.dependencies_dict = {
            "project" : [],
            "seed" : [],
            "generation" : []
        } ## List of cell objects to update when a key change changes. New dependencies
        ## may be added with new functions.
        self.seed = None
        self.generation = None
        
        self.net = None
        self.type = None ## Ex of type: "pareto"
        self.extra_variables = {} ## Allows to add new variables with a new cell
        ## object
        self.select_project = Select_Project(self)
        self.select_seed = Select_Seed(self)
        self.plot_pareto_fronts = Plot_Pareto_Fronts(self)
        self.plot_evolution_observable = Plot_Evolution_Observable(self)
        self.select_generation = Select_Generation(self)
        self.plot_layout = Plot_Layout(self)
        self.delete_nodes = Delete_Nodes(self)
        self.run_dynamics = Run_Dynamics(self)
        self.plot_dynamics = Plot_Dynamics(self)
###### MODIF HERE ######
        self.plot_sizediff = Plot_SizeDiff(self)
        self.plot_whi5sizecorrelations = Plot_Whi5SizeCorrelations(self)
        self.plot_dynamicsqty = Plot_DynamicsQty(self)
        self.plot_sizedist = Plot_SizeDistributions(self)
###### MODIF HERE ######
        self.plot_cell_profile = Plot_Cell_Profile(self)
        self.save_network = Save_Network(self)
    def get_net(self):
        return self.net

class CellModule():
    """
    Template class from which a module should inheritate.
    """
    def __init__(self,Notebook):
        self.notebook = Notebook
    def display(self):
        raise NotImplemented("Please define a display function for your module.")
    def update(self):
        raise NotImplemented("Please define a update function for your module.")

class Select_Project(CellModule):
        def __init__(self,Notebook):
            super(Select_Project, self).__init__(Notebook)
            self.widget_select_project = widgets.Text(value='',placeholder='Name of project directory',description='Directory:',disabled=False)
            self.widget_loadDir = widgets.Button(description="Load Run",disabled=True)
            self.foundFile_widget = widgets.HTML("")

        def inspect_run(self,path):
            """
            Test if the dir name exists

            Args:
                path (str): path of the directory
            """
            self.update()
            if os.path.isdir(path) and not self.notebook.project:
                self.foundFile_widget.value = found_str
                self.widget_loadDir.disabled=False
            else:
                self.foundFile_widget.value = notfound_str
                self.widget_loadDir.disabled=True

        def load_project(self,button):
            """
            Load the project directory in a somulation object.

            Args:
                directory : Path of the project
            """
            self.notebook.sim = Simulation(self.widget_select_project.value)
            self.notebook.project = self.widget_select_project.value
            self.widget_loadDir.button_style = "success"
            self.widget_loadDir.disabled=True
            self.notebook.type = self.notebook.sim.type
            print("To load a new project, please restart the kernel")
            print("Kernel > Restart & Run All")
            for cell in self.notebook.dependencies_dict["project"]:
                cell.update()


        def display(self):
            self.widget_loadDir.on_click(self.load_project)
            interactive(self.inspect_run,path=self.widget_select_project);
            main_options = widgets.VBox([widgets.HBox([self.widget_select_project,self.foundFile_widget]),self.widget_loadDir])
            display(main_options)
            #display(self.widget_select_project)

        def update(self):
            """
                Clears what needs to be cleared when the directory is changed.
            """
            self.widget_loadDir.disabled = True
            self.widget_loadDir.button_style = ""

class Select_Seed(CellModule):
    def __init__(self,Notebook):
        super(Select_Seed, self).__init__(Notebook)
        self.notebook = Notebook
        self.widget_select_seed = widgets.Dropdown(options={"None":None},value=None,description='Seed:',disabled=True)
        self.notebook.dependencies_dict["project"].append(self)

    def read_seed(self,seed_name):
        self.notebook.seed =  seed_name
        for cell in self.notebook.dependencies_dict["seed"]:
            cell.update()

    def display(self):
        interactive(self.read_seed,seed_name=self.widget_select_seed)
        display(self.widget_select_seed)

    def update(self):
        if self.notebook.project is None:
            self.notebook.seed = None
            self.widget_select_seed.options = {"None":None}
            self.widget_select_seed.disabled = True
            self.widget_select_seed.value = None
        else:
            self.widget_select_seed.disabled = False
            self.widget_select_seed.options = {"Seed {}".format(i):i for i,seed in self.notebook.sim.seeds.items()}
            self.widget_select_seed.value = self.widget_select_seed.options[list(self.widget_select_seed.options.keys())[0]]
            self.notebook.seed = self.widget_select_seed.value

class Plot_Evolution_Observable(CellModule):
    def __init__(self,Notebook):
        super(Plot_Evolution_Observable, self).__init__(Notebook)
        self.widget_Xobs = widgets.Dropdown(options=[None],value=None,description='x-axis:',disabled=True)
        self.widget_Yobs = widgets.Dropdown(options=[None],value=None,description='y-axis:',disabled=True)
        self.widget_replot_observable = widgets.Button(description="Plot",disabled=True)
        self.notebook.dependencies_dict["seed"].append(self)
    def replot_observable(self,b):
        
        plt.close()
        clear_output()
        display(self.plot_observable_options)
        self.notebook.sim.custom_plot(self.notebook.seed,self.widget_Xobs.value,[self.widget_Yobs.value])


    def display(self):
        self.widget_replot_observable.on_click(self.replot_observable)
        self.plot_observable_options = widgets.VBox([widgets.HBox([self.widget_Xobs,self.widget_Yobs]),widgets.HBox([self.widget_replot_observable])])
        display(self.plot_observable_options)

    def update(self):
        if self.notebook.seed is None:
            self.widget_Xobs.disabled = self.widget_Yobs.disabled = self.widget_replot_observable.disabled = True
            self.widget_Xobs.value = self.widget_Yobs.value = None
        else:
            self.widget_Xobs.disabled = self.widget_Yobs.disabled = self.widget_replot_observable.disabled = False
            self.widget_Xobs.options = list(self.notebook.sim.seeds[self.notebook.seed].observables.keys())
            self.widget_Yobs.options = list(self.widget_Xobs.options)
            self.widget_Xobs.value = "generation"
            self.widget_Yobs.value = self.notebook.sim.seeds[self.notebook.seed].default_observable

class Select_Generation(CellModule):
    def __init__(self,Notebook):
        super(Select_Generation, self).__init__(Notebook)
        self.notebook.dependencies_dict["seed"].append(self)
        self.widget_gen = widgets.IntSlider(value = 0,min=0,max=0,description = 'Gen:',disabled=True)
        self.widget_restart_gen = widgets.IntSlider(value = 0,min=0,max=0,description = 'Restart Gen:',disabled=True)
        self.widget_restart_net = widgets.IntSlider(value = 0,min=0,max=0,description = 'Network:',disabled=True)
    def read_best_generation(self,gen_index):
        if not self.widget_gen.disabled:
            self.notebook.generation = gen_index
            self.notebook.net = self.notebook.sim.get_best_net(self.notebook.seed,self.notebook.generation)
            for cell in self.notebook.dependencies_dict["generation"]:
                cell.update()
    def read_restart_generation(self,gen_index,net_index):
        if not self.widget_restart_gen.disabled:
            self.notebook.generation = gen_index
            self.notebook.net = self.notebook.sim.get_backup_net(self.notebook.seed,gen_index,net_index)
            for cell in self.notebook.dependencies_dict["generation"]:
                cell.update()
    def display(self):
        if not hasattr(self,"to_display"):
            select1 = interactive(self.read_best_generation,gen_index=self.widget_gen)
            select2 = interactive(self.read_restart_generation,gen_index=self.widget_restart_gen,net_index=self.widget_restart_net)
            widget1 = widgets.VBox([widgets.HTML("Select Best Network"),select1])
            widget2 = widgets.VBox([widgets.HTML("Select Backup Network"),select2])
            self.to_display = widgets.HBox([widget1,widget2])
        display(self.to_display)

    def update(self):
        if self.notebook.seed is None:
            self.widget_gen.value = 0
            self.widget_gen.min = self.widget_gen.max = 0
            self.widget_gen.disabled = True
            self.widget_restart_gen.disabled = True
            self.widget_restart_net.disabled = True
            self.notebook.generation = None
            self.notebook.net = None
        else:
            self.notebook.generation = None
            self.notebook.net = None
            self.widget_gen.value = 0
            self.widget_gen.disabled = False
            self.widget_gen.max = len(self.notebook.sim.seeds[self.notebook.seed].generations)-1
            self.widget_restart_gen.disabled = False
            restart_generations = list(self.notebook.sim.seeds[self.notebook.seed].restart_generations)
            step = restart_generations[1]
            self.widget_restart_gen.max = restart_generations[-1]
            self.widget_restart_gen.step = step
            self.widget_restart_net.disabled = False
            self.widget_restart_net.max = self.notebook.sim.seeds[self.notebook.seed].pop_size -1

class Plot_Layout(CellModule):
    def __init__(self,Notebook):
        super(Plot_Layout, self).__init__(Notebook)
        self.notebook.dependencies_dict["seed"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.button_plotLayout = widgets.Button(description="Plot network layout",disabled=True)

    def plot_layout(self,button):
        plt.close()
        clear_output()
        self.display()
        self.notebook.net.draw()

    def update(self):
        if self.notebook.generation is None:
            self.button_plotLayout.disabled = True
        else:
            self.button_plotLayout.disabled = False

    def display(self):
        self.button_plotLayout.on_click(self.plot_layout)
        display(self.button_plotLayout)

class Save_Network(CellModule):
    def __init__(self,Notebook):
        super(Save_Network, self).__init__(Notebook)
        self.notebook.dependencies_dict["seed"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_filename = widgets.Text(value='my_network.net',placeholder='Name of network',description='file.net',disabled=False)
        self.button_saveNetwork = widgets.Button(description="Save network",disabled=True)

    def save_network(self,button):
        self.notebook.net.store_to_pickle(self.widget_filename.value)

    def update(self):
        if self.notebook.generation is None:
            self.button_saveNetwork.disabled = True
        else:
            self.button_saveNetwork.disabled = False

    def display(self):
        self.button_saveNetwork.on_click(self.save_network)
        main_options = widgets.VBox([self.widget_filename,self.button_saveNetwork])
        display(main_options)

class Run_Dynamics(CellModule):
    def __init__(self,Notebook):
        super(Run_Dynamics, self).__init__(Notebook)
        self.widget_nputs = widgets.IntText(value=1,description='N :',disabled=False)
        self.button_launchRun = widgets.Button(description="Run dynamics",disabled=True)
        self.notebook.dependencies_dict["generation"].append(self)
        self.notebook.dependencies_dict["dynamics"] = []
        self.notebook.extra_variables = {"ntries":None}
    def launch_dynamics(self,button):
        self.notebook.sim.run_dynamics(net=self.notebook.net,erase_buffer=False,trial=self.widget_nputs.value)
        self.notebook.extra_variables["ntries"] = self.widget_nputs.value
        for cell in self.notebook.dependencies_dict["dynamics"]:
            cell.update()
    def update(self):
        if self.notebook.generation is None:
            self.button_launchRun.disabled = True
            self.notebook.sim.buffer_data = None
            self.notebook.extra_variables["ntries"] = None
        else:
            self.notebook.extra_variables["ntries"] = None
            self.notebook.sim.buffer_data = None
            self.button_launchRun.disabled = False
        for cell in self.notebook.dependencies_dict["dynamics"]:
            cell.update()

    def display(self):
        self.button_launchRun.on_click(self.launch_dynamics)
        display(widgets.HBox([self.widget_nputs,self.button_launchRun]))

        ################################
    ##############MODIF HERE################
        ################################

class Plot_SizeDiff(CellModule):
    def __init__(self,Notebook):
        super(Plot_SizeDiff,self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectCell = widgets.IntSlider(value = 0,min=0,max=0,description = 'Cell:',disabled=True)
        self.button_plotsizediff = widgets.Button(description="Plot size difference",disabled=True)

    def plot_sizediff(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_SizeDiff(self.widget_selectInput.value,cell=self.widget_selectCell.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectCell.value=self.widget_selectCell.min=self.widget_selectCell.max = 0
            self.button_plotsizediff.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectCell.max = self.notebook.sim.inits.prmt["ncelltot"]-1
            self.widget_selectInput.disabled = self.widget_selectCell.disabled = False
            self.button_plotsizediff.disabled = False

    def display(self):
        self.button_plotsizediff.on_click(self.plot_sizediff)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectCell,self.button_plotsizediff])
        display(self.widget)    

class Plot_SizeDistributions(CellModule):
    def __init__(self,Notebook):
        super(Plot_SizeDistributions,self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectCell = widgets.IntSlider(value = 0,min=0,max=0,description = 'Cell:',disabled=True)
        self.button_plotsizedist = widgets.Button(description="Size distributions",disabled=True)

    def plot_sizedist(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_SizeDistributions(self.widget_selectInput.value,cell=self.widget_selectCell.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectCell.value=self.widget_selectCell.min=self.widget_selectCell.max = 0
            self.button_plotsizedist.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectCell.max = self.notebook.sim.inits.prmt["ncelltot"]-1
            self.widget_selectInput.disabled = self.widget_selectCell.disabled = False
            self.button_plotsizedist.disabled = False

    def display(self):
        self.button_plotsizedist.on_click(self.plot_sizedist)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectCell,self.button_plotsizedist])
        display(self.widget)

class Plot_Whi5SizeCorrelations(CellModule):
    def __init__(self,Notebook):
        super(Plot_Whi5SizeCorrelations,self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectCell = widgets.IntSlider(value = 0,min=0,max=0,description = 'Cell:',disabled=True)
        self.button_plotwhi5sizecorrelations = widgets.Button(description="Whi5 & Size",disabled=True)

    def plot_whi5sizecorrelations(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_Whi5SizeCorrelations(self.widget_selectInput.value,cell=self.widget_selectCell.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectCell.value=self.widget_selectCell.min=self.widget_selectCell.max = 0
            self.button_plotwhi5sizecorrelations.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectCell.max = self.notebook.sim.inits.prmt["ncelltot"]-1
            self.widget_selectInput.disabled = self.widget_selectCell.disabled = False
            self.button_plotwhi5sizecorrelations.disabled = False

    def display(self):
        self.button_plotwhi5sizecorrelations.on_click(self.plot_whi5sizecorrelations)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectCell,self.button_plotwhi5sizecorrelations])
        display(self.widget)

class Plot_DynamicsQty(CellModule):
    def __init__(self,Notebook):
        super(Plot_DynamicsQty, self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectCell = widgets.IntSlider(value = 0,min=0,max=0,description = 'Cell:',disabled=True)
        self.button_plotdynamicsqty = widgets.Button(description="Plot quantities",disabled=True)

    def plot_dynamicsqty(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_TimeCourseQty(self.widget_selectInput.value,cell=self.widget_selectCell.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectCell.value=self.widget_selectCell.min=self.widget_selectCell.max = 0
            self.button_plotdynamicsqty.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectCell.max = self.notebook.sim.inits.prmt["ncelltot"]-1
            self.widget_selectInput.disabled = self.widget_selectCell.disabled = False
            self.button_plotdynamicsqty.disabled = False

    def display(self):
        self.button_plotdynamicsqty.on_click(self.plot_dynamicsqty)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectCell,self.button_plotdynamicsqty])
        display(self.widget)

        ################################
    ##############MODIF HERE################
        ################################ ^^^^

class Plot_Dynamics(CellModule):
    def __init__(self,Notebook):
        super(Plot_Dynamics, self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectCell = widgets.IntSlider(value = 0,min=0,max=0,description = 'Cell:',disabled=True)
        self.button_plotdynamics = widgets.Button(description="Plot dynamics",disabled=True)

    def plot_dynamics(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_TimeCourse(self.widget_selectInput.value,cell=self.widget_selectCell.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectCell.value=self.widget_selectCell.min=self.widget_selectCell.max = 0
            self.button_plotdynamics.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectCell.max = self.notebook.sim.inits.prmt["ncelltot"]-1
            self.widget_selectInput.disabled = self.widget_selectCell.disabled = False
            self.button_plotdynamics.disabled = False

    def display(self):
        self.button_plotdynamics.on_click(self.plot_dynamics)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectCell,self.button_plotdynamics])
        display(self.widget)

class Plot_Cell_Profile(CellModule):
    def __init__(self,Notebook):
        super(Plot_Cell_Profile, self).__init__(Notebook)
        self.notebook.dependencies_dict["dynamics"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.widget_selectInput = widgets.IntSlider(value = 0,min=0,max=0,description = 'Input:',disabled=True)
        self.widget_selectTime = widgets.IntSlider(value = 0,min=0,max=0,description = 'Time:',disabled=True)
        self.button_plotdynamics = widgets.Button(description="Plot profile",disabled=True)

    def plot_dynamics(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        self.notebook.sim.Plot_Profile(trial_index=self.widget_selectInput.value,time=self.widget_selectTime.value)

    def update(self):
        if self.notebook.extra_variables.get("ntries",None) is None or self.notebook.generation is None:
            self.widget_selectInput.value=self.widget_selectInput.min=self.widget_selectInput.max = 0
            self.widget_selectTime.value=self.widget_selectTime.min=self.widget_selectTime.max = 0
            self.button_plotdynamics.disabled = True
        else:
            self.widget_selectInput.max = self.notebook.extra_variables["ntries"]-1
            self.widget_selectTime.max = self.notebook.sim.inits.prmt["nstep"]-1
            self.widget_selectInput.disabled = self.widget_selectTime.disabled = False
            self.button_plotdynamics.disabled = False

    def display(self):
        self.button_plotdynamics.on_click(self.plot_dynamics)
        self.widget = widgets.HBox([self.widget_selectInput,self.widget_selectTime,self.button_plotdynamics])
        display(self.widget)

class Plot_Pareto_Fronts(CellModule):
    def __init__(self,Notebook):
        super(Plot_Pareto_Fronts, self).__init__(Notebook)
        self.notebook.dependencies_dict["seed"].append(self)
        self.widget_selectGenerations = widgets.SelectMultiple(options=[None],value=[None],description='Generation',disabled=True)
        self.widget_selectText = widgets.Text(value="",placeholder="List of generations separated with commas.",disabled=True)
        self.widget_plot = widgets.Button(description="Plot Pareto Fronts",disabled=True)
        self._widget_with_indexes  = widgets.Checkbox(value=False,description='Display indexes',disabled=False)


    def plot_function(self,button):
        plt.close()
        clear_output()
        display(self.widget)
        gen = self.widget_selectGenerations.value
        if self.widget_selectText.value:
            gen = [int(xx) for xx in self.widget_selectText.value.split(",")]
            #gen = [int(xx) for xx in self.widget_selectText.value.split(",")]

        self.notebook.sim.seeds[self.notebook.seed].plot_pareto_fronts(gen,max_rank=3)#self._widget_with_indexes.value)
    def update(self):
        if self.notebook.seed is None or self.notebook.type!="pareto":
            self.widget_selectGenerations.options = [None]
            self.widget_selectGenerations.value = [None]
            self.widget_selectGenerations.disabled = True
            self.widget_selectText.value = ""
            self.widget_selectText.disabled = True
            self.widget_plot.disabled = True
            self._widget_with_indexes.value = False
        else:
            self._widget_with_indexes.value = False
            self.widget_selectGenerations.disabled = False
            self.widget_plot.disabled = False
            self.widget_selectGenerations.options = self.notebook.sim.seeds[self.notebook.seed].restart_generations
            self.widget_selectGenerations.value = []
            self.widget_selectText.value = ""
            self.widget_selectText.disabled = False

    def display(self):

        #interactive(self.read_selected,generations=self.widget_selectGenerations)
        self.widget_plot.on_click(self.plot_function)
        instructions  = widgets.HTML("<p>Press <i>ctrl</i>, <i>cmd</i>, or <i>shift</i>  for multi-select</p>")
        self.widget = widgets.VBox([instructions,widgets.HBox([self.widget_selectGenerations,self.widget_selectText]),self.widget_plot])
        #to_display = widgets.VBox([self.widget_plot])
        display(self.widget)

def get_interactions(net):
    """
    Returns a dictionary of the interactions that are not CorePromoters.
    The key of the dictionary is a formated string containing the species of the interactions.
    """
    net.write_id()
    def find_species(func):
        parents = func(inter)
        all_species = False
        while not all_species:
            new_parents = []
            all_species = True
            #import pdb;pdb.set_trace()
            for pp in parents:

                if type(pp) is phievo.Networks.classes_eds2.Species:

                    new_parents += [pp]
                else:

                    new_parents += func(pp)
                    all_species = False
            parents = new_parents
        return parents
    inter_dict = {}
    for inter in net.dict_types.get("Interaction",[]):
        inter_type = re.search("\.(\w+)'",str(type(inter))).group(1)
        if inter_type == "CorePromoter":
            continue
        predecessors = find_species(net.graph.list_predecessors)
        sucessors = find_species(net.graph.list_successors)

        to_str = lambda species: ",".join([re.sub("\[|\]","", ss.id) for ss in species])
        inter_key = " ".join([inter_type,"(",to_str(predecessors),"->",to_str(sucessors),")"])
        inter_dict[inter_key] = inter
    return inter_dict

def get_species(net):
        return {re.sub("\[|\]","", ss.id):ss for ss in net.dict_types["Species"]}


class Delete_Nodes(CellModule):
    def __init__(self,Notebook):
        super(Delete_Nodes, self).__init__(Notebook)
        self.notebook.dependencies_dict["seed"].append(self)
        self.notebook.dependencies_dict["generation"].append(self)
        self.select_s = widgets.Dropdown(description="Species:",options={})
        self.button_s = widgets.Button(description="Delete",button_style="danger")
        self.select_i = widgets.Dropdown(description="Interaction:",options={})
        self.button_i = widgets.Button(description="Delete",button_style="danger")

        self.widget_list = [self.select_s,self.button_s,self.select_i,self.button_i]
        for i in range(len(self.widget_list)):
            self.widget_list[i].disabled = True
    def plot_layout(self,button):
        plt.close()
        clear_output()
        self.notebook.net.draw()

    def update(self):
        if self.notebook.net is None:
            for i in range(len(self.widget_list)):
                self.widget_list[i].disabled = True
        else:
            for i in range(len(self.widget_list)):
                self.widget_list[i].disabled = False
            self.select_s.options = get_species(self.notebook.net)
            if len(self.select_s.options)==0:
                self.button_s.disabled = True
            else:
                self.select_s.value = list(self.select_s.options.values())[0]
                
            self.select_i.options = get_interactions(self.notebook.net)
            if len(self.select_i.options)==0:
                self.button_i.disabled = True
            else:
                self.select_i.value = list(self.select_i.options.values())[0]

    def delete_species(self,button):
        clear_output()
        self.display()
        if "Input" not in self.select_s.value.list_types():
            index = int(re.search("\d+",self.select_s.value.id).group(0))            
            self.notebook.net.delete_clean(index,target="species")
            self.notebook.net.write_id()
            self.update()
        else:
            print("Cannot delete Input species.")
    def delete_interaction(self,button):
        clear_output()
        self.display()
        
        index = int(re.search("\d+",self.select_i.value.id).group(0))
        self.notebook.net.delete_clean(index,target="interaction")
        self.notebook.net.write_id()
        self.update()
    def display(self,first_time=True):
        if first_time:
            self.button_s.on_click(self.delete_species)
            self.button_i.on_click(self.delete_interaction)
        display(widgets.VBox([widgets.HBox([self.select_s,self.button_s]),widgets.HBox([self.select_i,self.button_i])]))
