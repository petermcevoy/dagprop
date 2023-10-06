Small personal project.

Directed Acyclic Graph with n-dimensional arrays (tensors) and operators to describe computation.
The graph can be evaluated in a forward pass to give a result and also in a backward pass to perform
backpropagation of gradients.

```bash
$ zig build -Doptimize=ReleaseSafe example_simple
info: Performing forward pass in DAG
Value of sub node is: 1.8e+01
info: Performing backward pass in DAG
Backpropagated gradient of intial constant: 6.0e+00
```
