### Part A: Image Processing with Cellular Non-linear Networks [1 pt each, Q6 is 2 pts]

The `cnn.py` builds a cellular non-linear network for image processing problems, where an NxM cell cellular non-linear network
proesses an NxM image. The greyscale image is set with the `U` matrix, where 

You will be editing the `run_cnn()` function, which sets up the cellular non-linear network to process a random image from the 
MNIST dataset. The image is saved to `cnn_input.png`, and an animation of the CNN cell evolution over time is saved to `cnn.gif`.

Q1. Try the CNN with the A1, B1, and z1 parameters on the input image. What does the CNN do with the current parameter configuration? Does the result appear instantaneously?

 
----

Q2. Currently the cells initial values are all set to zero. How does the behavior change if they are randomly instantiated to a value between 0-0.1?


----

Q3. Next, enable `EXERCISE=2` this introduces 20% relative mismatch into the parameters. What happens when all three are perturbed -- do you still get the same result? Which of these parameters has the most significant impact when perturbed?




---

Q4. Now, enable `EXERCISE=3` which uses a different set of parameters A2, B2, and z2. What does this parameterization of the CNN do?


---


Q5. Now enable `EXERCISE=4` which introduces 20% relative mismatch to the parameters. What happens when all three are perturbed? How does this compare to 
when the template parameters A1, B1, z1 were mismatched.


---


Q6. Currently the CNN uses the clip function for saturation. What happens if your nonlinearity is imperfect, and actually implements a sigmoid? What happens? Do this by changing the `clip` function invocations to sigmoid function invocations and then rerunning some experiments. Make sure the sigmoid is properly centered so it performs the same operation as the clip function, but with smoother edges. 


---
Q7. Multiply the right-hand side of the differential equation by 10. This can be done by modifying the diffeqs function. What happens to the system? 


### Part B: Phase-Domain Oscillator-Based Computing [1pt/question]


We will next play with oscillator-based computing. I have written the code necessary to simulate the oscillator-based computing paradigm. The simulator
will generate the time-series plots of the phase over time, the frequency over time, and an animation of the oscillators in the `images-obc` directory.

We will first experiment with oscillators that have the same natural frequency, but start off with random phases. The experimental setup is in the 
`simple_osc_phaseonly` function. The natural frequency of the oscillators is 1 khz or 1000 hz, and the phase is randomly instantiated to be between 0 and 2\pi. You can change the exercise you're doing by setting the `EXERCISE` variable.

Q1. Run the function unmodified -- this will simulate a set of free-running oscillators. How do the phase and the frequency change over time? Do the oscillators synchronize? Does the phase ever stop changing? How do frequency and phase relate?


---

Q1b. How do frequency and phase relate? How does frequency relate to the slope of the phase trajectories?

-----

Q2. Enable `EXERCISE=2`, which couples all the oscillators together. What happens to the frequency as the system evolves? What happens to the phase as the system evolves? How can you tell when the frequency is synchronized (look at `freq.png`), how can you tell when phase is synchronized (look at `phase.png`).


----

Q3. Enable `EXERCISE=3`, which couples all the oscillators together, but makes the coupling strength between oscillators 1/2 very weak. How does the phase evolution of this system compare to the phase evolution from the previous configuration? Do the oscillators all synchronize by the end of the simulation? 

----

Q4. Enable `EXERCISE=4`, which couples oscillators 0-1, and 2-3 together and negatively couples oscillators 1-2 together. How does the phase evolve over time in this problem setting? Which oscillators synchronize? Which oscillators do not?


### Part C: Frequency-Domain Oscillator-Based Computing with Non-idealities [1pt/question]

Next, we implement a modified version of the frequency-domain oscillator-based computing network from the "A Nanotechnology-Ready Computing Scheme based on a Weakly Coupled Oscillator Network" paper. This network has two "core" oscillators that operate at 5.6 and 5.8 Hz and one "input" oscillator A that can be set to anywhere between 5-6.8 Hz. The coupling strength between core oscillators is 0.4 and the coupling strength between the core and input oscillator is 1.2. The core oscillators will tend to synchronize when the input oscillator is in a certain frequency range. 

Q1. Enable `EXERCISE=1` -- this disconnects the input oscillator "A" from the network. What happens to the frequency of the core oscillators when this happens?


---

Q2. In Exercise 1, do the oscillators that synchronize in frequency also take on the same phase?


---

Q3. Enable `EXERCISE=2` -- this sets the frequency of oscillator "A" to a medium frequency (5.8) and reconnects input oscillator "A" to the system. What happens to the oscillator network? Which oscillators synchronize in frequency? Which oscillators synchronize in phase? 


----

Q4. Enable `EXERCISE=3` -- this sets the frequency of oscillator "A" to a low frequency (1.0) and reconnects oscillator "A". What happens to the oscillator network? Which oscillators synchronize in frequency? Which oscillators synchronize in phase?

----

Q5. Enable `EXERCISE=4` -- this sets the frequency of oscillator "A" to a high frequency (11.0). Again what happens?


----

Q6. Enable `EXERCISE=5` - -this sets the frequency of oscillator "A" to a low frequency that is closer to 5.6-5.8. What is the synchronization behavior you observe?





### Part D: Grade Optimization with Integer Linear Programming [6 pts]

Next we will use integer linear programming to optimize your grace days to maximize your grade. Look at `grade_optimizer.py`, which has the scaffold and an example grade report. The scaffold code has comments indicating where you should add constraints and modify expressions.

Q1. What are the variables in this constraint problem? Are they integer or real variables? [1 pt]

---

Q2. Fill in the constraints / update the necessary expressions and run the solver. Don't worry about handling the fact that we can't apply grace days to the final proposal deadline. What's the optimal assignment of grace days for the example grade report? What grade is achieved? [5 pts]





