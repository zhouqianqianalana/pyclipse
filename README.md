# pyclipse

"Pyclipse" is an automated Python-Eclipse package for static model construction, simulation execution, and result analysis, reducing workflow duration by over 80%.

![image](https://github.com/user-attachments/assets/77579e9c-1b25-435d-95bc-5dcdadfa6bb6)


This repository is designed to generate all the input files needed for running reservoir simulations in Eclipse, specifically for Gulf of Mexico Paleogene Reservoirs. The process begins with creating a geological lobe model, followed by populating the grids with specified properties. The package then runs all possible parameter combinations in parallel by executing Eclipse, efficiently monitoring CPU usage and automatically starting new simulations as soon as resources are available. This ensures that the CPU is utilized fully and efficiently.

![image](https://github.com/user-attachments/assets/0a87241e-599c-4fd8-b3c0-cdab0f33dcc1)
![image](https://github.com/user-attachments/assets/8b672e7e-d91f-449f-a21d-9ad031cefb2c)

The example scripts can be found in examples directory