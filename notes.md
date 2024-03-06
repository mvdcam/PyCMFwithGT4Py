
Function to optimize : 
- earth.py :
    - add_energy()

- ticking_earth.py :
    - average_temperature()
        - dependencies : 
            - neighbors, index
            - grid_chunk.temperature
            - grid_chunk.mass
            - grid_chunk.heat_transfer_coefficient
            - grid_chunk.specific_heat_capacity
            - grid_chunk.add_energy()

    - NEED TO ADD water_evaporation() from the grid_chunk module
        - dependencies : 
            - grid_chunk.temperature
            - grid_chunk.mass


to check : 
- ticking_sun.py
    - radiate_energy_outwards()



temperature ?    
 def __get_temperature(self):
        return self.energy / (self.specific_heat_capacity * self.mass)


        def constants : /Users/matthiasvandercam/Desktop/Informatique/Mémoire/pace-main/util/pace/util/constants.py