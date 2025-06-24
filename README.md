# Reducing Calls to the Simulator in SBI
This is a seminar work on the paper "[Reducing Calls to the Simulator in Simulation Based Inference](https://arxiv.org/abs/2504.11925)".
If you want to have a look at the code from the originial paper, please check out their [github](https://github.com/MaverickMeerkat/ReducingSimulatorCallsSBI).

## Running the code
To run the code, please make sure you have all the necessary dependencies installed; the dependencies listed by the authors of the original paper are in the `requirements.txt` file. To install the requirements, use the following command:
```
pip install -r requirements.txt
```
The code of all the idividual tasks that the paper provides are also found here in the directory `tasks`. However, in my short work on this, I only worked on the task `TwoMoons.py`, which I copied over into my file `demo.py`.
I compiled my own demonstration version of the code in the file `demo.py`, and you can run it with the following comand:
```
python demo.py
```
In this demonstration version, I only had a look at a small subsection of the code from the original paper, a lot from the code is copied over from the original paper modulo some restructurings and renamings.
With this small part of the code you can partially reproduce the results of the `mmd` and `c2st` mean and variances for the two moons task with a budget of 200, both for the surrogate method and for the support point method and also the combined version.
I only focused on one task and one budget because of my limited time and compuational resources.
The code for the calculation of the support points is in the file `SupportPoints.py`. The code here is also largely copied from the original paper but has also been restructured and variables have been renamed.