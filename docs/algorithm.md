# Algorithm

## Overview

`pylsci` computes a **Least Squares Reference Circle (LSCI)** from a set of two-dimensional points and evaluates the corresponding roundness.

Given input points

$$
x=(x_1,\ldots,x_n),\qquad
y=(y_1,\ldots,y_n),
$$

`pylsci` computes

- fitted circle center
- fitted circle radius
- roundness

using a linear least-squares formulation
and returns a `FittedCircle` object.

## Mathematical formulation

### Circle equation

A circle is represented by

$$
(x-a)^2 + (y-b)^2 = R^2,
$$

where

- $a$ : center x-coordinate
- $b$ : center y-coordinate
- $R$ : radius

Expanding,

$$
x^2 + y^2 - 2ax - 2by - (R^2 - a^2 - b^2) = 0.
$$

Introducing the auxiliary parameters makes the problem linear with respect to the unknowns.

$$
A:=2a,\qquad
B:=2b,\qquad
C:=R^2 - a^2 - b^2.
$$

The equation becomes

$$
x^2 + y^2 - Ax - By - C = 0.
$$

Unknown parameters are therefore

$$
\theta:={\begin{bmatrix} A & B & C \end{bmatrix}}^{\top}.
$$

## Least-squares formulation

For measured data, the points generally do not lie exactly on a circle.
Therefore,

$$
\Delta_i := {(x_i)}^2 + {(y_i)}^2 - A x_i - B y_i - C
$$

is generally nonzero.

The least-squares problem is to find

$$
\mathop{\text{argmin}}_\theta \sum_{i=1}^n {(\Delta_i)}^2
$$

### Derivation of the normal equation

Differentiating the objective function with respect to
$A,B,C$
and setting the derivatives to zero yields the normal equation.

```math
\begin{aligned}
\frac{\partial}{\partial A} \sum_{i=1}^n {(\Delta_i)}^2
&=
\sum_{i=1}^n \frac{\partial}{\partial A} \left\lbrace {(\Delta_i)}^2 \right\rbrace
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot \frac{\partial \Delta_i}{\partial A}
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot (- x_i)
\\ &=
-2 \sum_{i=1}^n \left\lbrace {(x_i)}^3 + x_i {(y_i)}^2 - A {(x_i)}^2 - B x_i y_i - C x_i \right\rbrace
\\ 
\\ 
\frac{\partial}{\partial B} \sum_{i=1}^n {(\Delta_i)}^2
&=
\sum_{i=1}^n \frac{\partial}{\partial B} \left\lbrace {(\Delta_i)}^2 \right\rbrace
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot \frac{\partial \Delta_i}{\partial B}
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot (- y_i)
\\ &=
-2 \sum_{i=1}^n \left\lbrace {(x_i)}^2 y_i + {(y_i)}^3 - A x_i y_i - B {(y_i)}^2 - C y_i \right\rbrace
\\ 
\\ 
\frac{\partial}{\partial C} \sum_{i=1}^n {(\Delta_i)}^2
&=
\sum_{i=1}^n \frac{\partial}{\partial C} \left\lbrace {(\Delta_i)}^2 \right\rbrace
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot \frac{\partial \Delta_i}{\partial C}
\\ &=
\sum_{i=1}^n (2 \Delta_i) \cdot (- 1)
\\ &=
-2 \sum_{i=1}^n \left\lbrace {(x_i)}^2 + {(y_i)}^2 - A x_i - B y_i - C \right\rbrace
\end{aligned}
```

Therefore

```math
\begin{aligned}
  &
  \begin{bmatrix}
  \frac{\partial}{\partial A} \sum_{i=1}^n {(\Delta_i)}^2 \\
  \frac{\partial}{\partial B} \sum_{i=1}^n {(\Delta_i)}^2 \\
  \frac{\partial}{\partial C} \sum_{i=1}^n {(\Delta_i)}^2
  \end{bmatrix}
  =
  \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
  \\ & \iff
  \sum_{i=1}^n \begin{bmatrix}
  {(x_i)}^3     + x_i {(y_i)}^2 - A {(x_i)}^2 - B x_i y_i   - C x_i \\
  {(x_i)}^2 y_i + {(y_i)}^3     - A x_i y_i   - B {(y_i)}^2 - C y_i \\
  {(x_i)}^2     + {(y_i)}^2     - A x_i       - B y_i       - C
  \end{bmatrix}
  =
  \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
  \\ & \iff
  \left(
  \sum_{i=1}^n \begin{bmatrix}
  {(x_i)}^2 & x_i y_i   & x_i \\
  x_i y_i   & {(y_i)}^2 & y_i \\
  x_i       & y_i       & 1
  \end{bmatrix}
  \right)
  \begin{bmatrix} A \\ B \\ C \end{bmatrix}
  =
  \sum_{i=1}^n \begin{bmatrix}
  {(x_i)}^3     + x_i {(y_i)}^2 \\
  {(x_i)}^2 y_i +     {(y_i)}^3 \\
  {(x_i)}^2     +     {(y_i)}^2
  \end{bmatrix}
  \\ & \iff
  \left(
  \sum_{i=1}^n \begin{bmatrix}
  {(x_i)}^2 & x_i y_i   & x_i \\
  x_i y_i   & {(y_i)}^2 & y_i \\
  x_i       & y_i       & 1
  \end{bmatrix}
  \right)
  \begin{bmatrix} A \\ B \\ C \end{bmatrix}
  =
  \sum_{i=1}^n \begin{bmatrix}
  x_i \left\lbrace {(x_i)}^2 + {(y_i)}^2 \right\rbrace \\
  y_i \left\lbrace {(x_i)}^2 + {(y_i)}^2 \right\rbrace \\
         {(x_i)}^2 + {(y_i)}^2
  \end{bmatrix}
\end{aligned}
```

This is exactly the linear system assembled in the implementation.

### Solving the normal equation

The resulting linear system is solved using

* [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve)
* [`pytensor.linalg.solve`](https://pytensor.readthedocs.io/en/stable/library/xtensor/linalg.html#pytensor.xtensor.linalg.solve)

depending on the selected backend.

## Recovering the circle

The linear system solves for the auxiliary parameters
$A,B,C$, not the circle parameters directly.
Since
$A = 2a, B = 2b, C = \sqrt{R^2 - a^2 - b^2}$,
the center and radius are recovered as

$$
a=\frac{A}{2},\qquad
b=\frac{B}{2},\qquad
R=\sqrt{a^2+b^2+C}.
$$

## Roundness evaluation

For each measured point,

$$
\delta_i := \sqrt{(x_i-a)^2+(y_i-b)^2}
$$

Roundness is defined as

$$
\max_i \delta_i - \min_i \delta_i.
$$

Consequently,

- a perfect circle yields zero roundness,
- deviations from circularity increase the value.

## Numerical stabilization

To improve numerical conditioning,
the implementation first translates all points so that their centroid becomes the origin.
Translation does not change the fitted radius or roundness; only the coordinate system is shifted.
The fitted center is translated back after solving.

This preprocessing is not mathematically required.
The least-squares formulation remains valid without it.
However, centering the data reduces the magnitude of the coefficients in the normal equation,
which typically improves its numerical conditioning and reduces the effect of floating-point round-off when the coordinates are far from the origin.

The centroid is

$$
\bar{x} := \frac{1}{n} \sum x_i,\qquad
\bar{y} := \frac{1}{n} \sum y_i
$$

Offset coordinates are

$$
x_i^\prime := x_i-\bar{x},\qquad
y_i^\prime := y_i-\bar{y}
$$

The normal equation is constructed using
$ (x_i^\prime , y_i^\prime) $

After solving, the estimated center is translated back,

$$
a=a^\prime + \bar{x},\qquad
b=b^\prime + \bar{y}.
$$

This restores the circle center to the original coordinate system.

## Computational complexity

The algorithm has linear complexity in the number of input points.
The dominant cost is evaluating several sums over the input,
followed by solving a fixed-size $ 3\times3 $ linear system.

## Backend differences

- The NumPy backend performs all computations eagerly.
- The PyTensor backend constructs symbolic expressions that can be evaluated later.

## References

- ISO 12181-1: Geometrical product specifications (GPS) — Roundness
- ISO 12181-2: Specification operators
