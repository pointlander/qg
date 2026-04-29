# Quantum Gravity
## Theory
```
kinetic energy
(1/2)*V^2*m = E
drop (1/2)
V^2*m = E

E = m*c^2
assume c = 1
E = m

substitute
V^2*E = m*c^2
V^2*E = E*c^2
V^2*E = E*V^2

gravitational potential energy
E = -G*M*m/r
assume 1kg point masses
E = -G/r
compute an adjacency matrix of the inverse distance between all point masses
E = -G.R
'.' is the hadamard product

V^2*(-G.R) = (-G.R)*V^2
V^2*(G.R) = (G.R)*V^2
```
R is a known real matrix, and V and G are complex matrices solved with backpropagation gradient descent.
V is then used to update the point particle positions and R.
## Discussion
The above equation may describe a multiverse.
If the `V^2` is dropped from one side, the equation makes for a strong clustering algorithm.
The overall form of the equation uses Heisenberg's non-commutative: A*B != B*A.
In general the equation computes truth that contradicts itself.
## Simulation
![point masses](verse.gif?raw=true)
## G calculation
![G](Gavg.png?raw=true)
## Learning loss
![loss](loss.png?raw=true)
