# Reducing Calls to the Simulator in SBI
This i a seminar work on the paper "Reducing Calls to the Simulator in Simulation Based Inference" [Link](https://arxiv.org/abs/2504.11925)
If you want to have a look at the code from the originial paper please check out there [github](https://github.com/MaverickMeerkat/ReducingSimulatorCallsSBI).

## Running the code
To run the code please make sure you have all the necessary dependencies installed, the are the dependencies from the original paper in the `requirements.txt` file. to install the use the following command:
```
pip install -r requirements.txt
```
The code of all the idividual tasks the paper provided are also found here in the folder `tasks`. However in my short work on this I only worked on the task `TwoMoons.py`.
I compiled my own demonstration version of the code in the file `demo.py`, and you can run it with the following comand:
```
python demo.py
```
In this demonstration version I only hade a look at a small subsection of the code from the original paper, a lot from the code is coppied over from the original paper minus some restructurings and renamings.
With this small part of the code oyu can reproduce the results of the `mmd` and `c2st` mean and variances for the two moons task with a budget of 200, both for the surrogate method and for the support point method. 
I have not included the code for the combined version here because the runtime of this was already quite long.
The code for the calculation of the support points is in the file `SupportPoints.py` The code here is also largely coppied from the original paper but has also been restructured and variables have been renamed.