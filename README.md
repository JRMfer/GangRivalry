# GangRivalry

## Introduction

In  the  USA,  rivalry between street gangs is a leading cause of homocides and other violence. In this model, based on the model of [Hegemann *et al.* (2011)](https://doi.org/10.1016/j.physa.2011.05.040), several walking strategies and geographical features of Hollenback, LA, are combined to possibly predict rivalry between these gangs. 
Three walking methods are investigated using for models: brownian motion and semi-biased Lévy walk with and without using the gang size in the bias. The model is built in the framework of Mesa.


## Prior requirements

Code is written in python 3.7.

In order to run the code Mesa should be installed
```
pip install -e git+https://github.com/projectmesa/mesa#egg=mesa
```



## Folder structure

In the main you find the file main.py in which you can run the model. This is supported by serveral files in the folders data, sensitiviy_analysis and code.

### Data

The folder data contains parameter settings and data that are necessary to run the model, including our estimation of the gang sizes, location of the gangs and road density and boundaries of the main areas of Hollenbeck. In addition basic parameter settings for the model are in this folder (minimal and maximal jump length, etc.).

### Sensitivity_analysis

In order to run the sensitivity analysis, the .npy files should be stored in the folders of the walking method. When sensitive.py is run in this folder plots are created of the accuracy measures ACC, F1 and MCC at different threshold values of edge creation, as well as for the node density, the nodal variance degree and the centrality.

### Classes
In the folder code, our main code is saved with three subfolders: classes, helpers and visualization.

Classes contains agents.py, configurations.py, model.py and schedule.py.

In agents.py contains the agents classes with their unique walking methods and a general gang class which contains general information of each gang.
In configuration.py contains a class with methods that can be used to load the data.
In model.py the model is described which is generalized for each walking method.
In schedule.py the schedule of the order in which the agents move is defined.

### Helpers

The folder helpers contains glasbey.py and helpers.py

In glasbey.py is used for color generation of the boundaries ([github](https://github.com/taketwo/glasbey)).
In helpers.py methods are described that help to get parameters from the model and support running of the model.

### Visualization

Visualization contains visualizer.py.

Visualizer.py contains methods to make plots of the data collected.




## Running the model

In order to run the model, you have to specify:

* Which walking method you want to use (brownian motion (BM), semi-biased Lévy walk (SBLN) or semi-biased Lévy walk with gravity with bigger gangs (GRAV))
* Number of times to run the model
* Number of iterations in the model (note: one iteration is moving one agent)
* After how many iterations you want to collect data
* File of the input data
* Name of the folder in which you want to save the data
* From which number you want to number your output data

So when the input is this:

```
python main.py BM 5 10000000 10000 input_data.txt MIKE 0

```

you run the brownian motion model 5 times with 10 milion steps and every 10 thousend steps the data is collected. Input comes from input_data.txt (recommended). Output is written in the folder results_MIKE.

## Ouput of the model

The standard output of the model is a csv file containing accuracy measures and shape metrics per time step the datacollector collects data (including the first step). The accuracy contains a tupe (ACC, F1, MCC) and the shape metrics contain anothed tupe (graph density,variance nodel degree, centrality).

In a .npy file is saved with a matrix that contains the total number of interactions between the gangs.

### Acknowledgements

This project has been made for the course Agent-Based Modeling of the master Computational Science of the University of Amsterdam. The course was teached by Mike Lees. Most of the data and set up of the models come from the article of [Hegemann *et al.* (2011)](https://doi.org/10.1016/j.physa.2011.05.040). Thank you.

Keep in mind that a lot of assumptions have been made in this model as well as in the input data.
