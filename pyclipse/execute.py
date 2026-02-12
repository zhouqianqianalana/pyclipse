import numpy as np
import subprocess, os, sys
import concurrent.futures
from pathlib import Path
from pyclipse.reservoir import Reservoir
from copy import deepcopy


class Executor:

    def __init__(self, datafile_path):
        self.datafile_path = datafile_path

    def run(self):
        subprocess.run(["eclrun", "eclipse", str(self.datafile_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



class Designer:

    def __init__(self, res, datafile_path, design_dirpath, max_runs_in_parallel, design_data, design_type='OFAT'):

        self.layers = deepcopy(res.layers)
        self.datafile_path = datafile_path
        self.datafile_name = self.datafile_path.name
        
        self.design_dirpath = Path(design_dirpath)
        if not self.design_dirpath.exists():
            self.design_dirpath.mkdir(parents=True)

        self.max_runs_in_parallel = max_runs_in_parallel
        self.design_data = design_data
        self.design_type = design_type
        self.features_list = list(self.design_data.keys())
        self.n_features = len(self.design_data.keys())


    def design_and_run(self):
        self.design()
        self.run()


    def design(self):
        if self.design_type == 'OFAT':
            self.design_sequential_loops()
        elif self.design_type == 'Factorial':
            self.design_nested_loops(self.features_list)


    def run(self):
        if self.design_type == 'OFAT':
            self.run_sequential_loops()
        elif self.design_type == 'Factorial':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.run_nested_loops(self.features_list, futures=[], executor=executor)
    

    def design_sequential_loops(self):
        for feature in self.features_list:
            feature_name, layer_number = feature.split('_')
            for value in self.design_data[feature]:
                self.layers_new = deepcopy(self.layers)
                self.layers_new[layer_number-1].set_property(feature_name,value)
                self.layers_new[layer_number-1].create_geology()
                self.res = Reservoir(self.layers_new)
                write_path = self.design_dirpath / (feature + '_' + str(value))
                self.res.write_eclipse_files(write_path,self.datafile_path)

    
    def design_nested_loops(self, temp_features_list, values_list=[]):
        if len(values_list) == self.n_features:
            self.layers_new = deepcopy(self.layers)
            for i,feature in enumerate(self.features_list):
                feature_name, layer_number = feature.split('_')
                self.layers_new[layer_number-1].set_property(feature_name, values_list[i])
                self.layers_new[layer_number-1].create_geology()
            self.res = Reservoir(self.layers_new)
            directory_names_list = [f"{s}_{f:.3f}" for s, f in zip(self.features_list, values_list)]
            write_path = self.design_dirpath.join_path(*directory_names_list)
            self.res.write_eclipse_files(write_path,self.datafile_path)
        else:
            for value in self.design_data[temp_features_list[0]]:
                self.design_nested_loops(temp_features_list[1:], values_list + [value])
    

    def run_sequential_loops(self):
        for feature in self.features_list:
            for value in self.design_data[feature]:
                datafile_path = self.design_dirpath.joinpath([feature + '_' + str(value),str(self.datafile_name)])
                subprocess.run(["eclrun", "eclipse", str(datafile_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def run_nested_loops(self, temp_features_list, values_list=[], futures=None, executor=None):
        if futures is None:
            futures = []
        
        if len(values_list) == self.n_features:
            directory_names_list = [f"{s}_{f:.3f}" for s, f in zip(self.features_list, values_list)]
            datafile_path = directory_names_list + [str(self.datafile_name)]
            datafile_path = self.design_dirpath.joinpath(*datafile_path)
            
            # Submit the job if under the limit
            if len(futures) < self.max_runs_in_parallel:
                futures.append(executor.submit(self.run_eclipse, datafile_path))
                print(f"Submitted Eclipse run in {datafile_path}")
            else:
                # Wait for the first job to complete if limit reached
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                # Remove the completed jobs from the futures list
                futures = [f for f in futures if not f.done()]
                # Submit new job after one completes
                futures.append(executor.submit(self.run_eclipse, datafile_path))
                print(f"Submitted Eclipse run in {datafile_path}")
        else:
            # Recursively process the next feature's possible values
            for value in self.design_data[temp_features_list[0]]:
                self.run_nested_loops(temp_features_list[1:], values_list + [value], futures, executor)

    
    def run_eclipse(self, datafile_path):
        subprocess.run(["eclrun", "eclipse", str(datafile_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

