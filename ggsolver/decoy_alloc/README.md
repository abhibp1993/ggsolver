# Description of CONFIG files

The simulation is run according to `.config` file saved in `configurations` folder. 
The components of `.config` are defined below. 

* `directory`: Directory where simulation output will be saved. Default is `<pwd>/out`. 
   If the directory is not empty and `overwrite` is `false`, an error will be raised.     
* `overwrite`: If `directory` exists, overwrite the existing files? Default `false`.    
* `type`: Type of experiment to run. Options: `enumeration`, `greedy`.
* `name`: Name of experiment. 
* `num_trials`: Number of times the experiment should be run. 
* `use_multiprocessing`: Use single-core or multi-core processing.
* `max_traps`: Number of traps available.
* `max_fakes`: Number of fake targets available.
* `graph`: Parameters using which the base game graph is generated.
    - `topology`: Topology of graph. Supported options `mesh, ring, star, tree` and `hybrid`. 
    - `nodes`: Number of nodes. 
    - `max_out_degree`: For applicable topologies, maximum out-degree of any node. Default `null`.   
    - `save`: Should the base game graph be saved?
    - `save_png`: Should the base game graph be saved in image format (PNG)?  
    - `name`: Name of file to use for saving. Default `<filename>_base`. 
  },
* `log`: Should the logs be saved. Default is `true`.
* `console_log_level`: Level of console logging output. Default is `info`. 
* `save_intermediate_solutions`: Whether intermediate solutions should be saved. Default is `false`
* `report`: List of graphs, reports etc. to generate. 
  Available options are `vod1_chart`, `state_char_graph`, `time_chart`, `space_chart`
