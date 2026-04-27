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

V^2*(-G.R) = (-G.R)*V^2
'.' is the hadamard product
V^2*(G.R) = (G.R)*V^2
```
R is known, and V and G are solved with backpropagation gradient descent.
V is then used to update the point particle positions and R.
## Simulation
![point masses](verse.gif?raw=true)
## G calculation
![G](Gavg.png?raw=true)
## Learning loss
![loss](loss.png?raw=true)
