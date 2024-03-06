# PyCMF

## Objectives

The objective of this project was to experiment with Object-Oriented technology in a Climate Modelling context. We used Climate Models as a basis on which to build a framework in OO to research the advantages and drawback of the use of Object-Oriented in a compute-intensive setting.

## Introduction

PYthon Climate Modelling Framework (or PyCMF for short) is a framework developped as part of my Master's thesis in Computer Science at
the ULB. The framework provides 3 layers of implementation :

- base_class (earth_base, grid_chunk_base, etc ...) and inheriting from BaseModel : Contains the basic
  structural/pythonic stuff for the class (correct inheritance, redefinition of dunder methods, etc ...)
- physical_class (earth, grid_chunk, etc ...) and inheriting from its base_class : Contains the physical properties (temperature, mass, etc ...) and
  method (behaviour for receiving electromagnetic radiation, etc ... ) for that class 
- ticking_class (ticking_earth, ticking_universe, etc ...) and inheriting from its physical_class as well as
  TickableModel, that is an interface to store all the class methods that have to be executed at each time step of the
  simulation via the `@TickableModel.on_tick` decorator

The framework is currently **not** able to provide accurate simulations of real-world physical process, but provides a
few examples with placeholder simulations such as the averaging of the temperature at each time step

## Running the code

### Required Libraries

- unittest for testing
- PyQt5 for the graphical interface
- numpy for the models


To run the framework, you can edit the script in `main.py` and then execute it with `python3.9 main.py`.

For running the program with the graphical interface, you can use the GUI command line argument and
execute `python3.9 main.py GUI`. In that way you will be able to dynamically change the simulation

## How to add a new model

To add a new model, you have to add between 1 and 3 files since the models are split in 3 different layers. First, create your $model.py file in physical_class and add all the physical properties of that model you need. Then, if necessary, create another file in base_class to handle all the pythonic behavior, such as iteration behavior, adding, substracting, memory use, data storage, saving the simulation, loading a simulation, etc ... Don't forget to make your second layer model inherit from the first layer model.

Finally, if you model has some variables that are updated over time, you will have to create a third file in ticking_class to define the different updates behavior with the `@TickingModel.on_tick(enabled=True)` decorator that you obtain by inheriting from TickingModel. Also, you must inherit from your second layer class to get all of your physical properties variables on which the update is done.

## How to add a new variable to the model

To enrich the framework, you can add new physical properties and all their associated methods for conveniance (getters, setters, etc ... ) in the second layer, physical_class. You can also change the `__init__` method of the class to allow for setting your variable when the model is built, in which case you will have to find the other use of that model and change the constructor's parameters as well.

Then, if your variable has a temporal dimension to it, you can add the temporal evolution in the third layer, ticking_class, where you can define a function decorated by `@TickingModel.on_tick(enabled=True)`


