Linear Algebra Tutorial
=======================

This tutorial demonstrates how to solve various linear algebra problems using the Math AI Agent. The system provides comprehensive support for matrix operations, eigenvalue problems, and decomposition methods.

Overview of Linear Algebra Features
------------------------------------

The Math AI Agent supports:

* **Basic Matrix Operations**: Addition, multiplication, determinant, inverse
* **Matrix Decompositions**: LU, QR, SVD (Singular Value Decomposition)
* **Eigenvalue Problems**: Eigenvalues, eigenvectors, diagonalization
* **Linear Systems**: Solving Ax = b using various methods
* **Matrix Analysis**: Rank, trace, condition number
* **Visualizations**: Matrix heatmaps, eigenvalue plots, decomposition displays

Basic Matrix Operations
-----------------------

Determinant Calculation
~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Calculate the determinant of a 2×2 matrix.

**Input:**
.. code-block:: text

   Calculate the determinant of the matrix: [[3, 2], [1, 4]]

**Expected Solution Process:**
1. Parse the matrix from the input
2. Apply determinant formula: det(A) = ad - bc
3. Calculate: (3)(4) - (2)(1) = 12 - 2 = 10
4. Verify the result through numerical methods

**What You'll See:**
- Step-by-step determinant calculation
- Formula explanation: det(A) = ad - bc
- Numerical verification
- Matrix visualization as a heatmap

Matrix Inverse
~~~~~~~~~~~~~~

**Problem:**
Find the inverse of a matrix and verify it.

**Input:**
.. code-block:: text

   Find the inverse of the matrix: [[2, 1], [1, 1]]

**Expected Solution Process:**
1. Check if matrix is invertible (det ≠ 0)
2. Calculate inverse using the formula or elimination
3. Verify that A × A⁻¹ = I (identity matrix)
4. Display both original and inverse matrices

**Advanced Example:**
.. code-block:: text

   Calculate the inverse of: [[1, 2, 3], [0, 1, 4], [5, 6, 0]]

**What You'll Learn:**
- When a matrix has an inverse
- Different methods for computing inverses
- Verification through matrix multiplication
- Numerical stability considerations

Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~

**Problem:**
Multiply two matrices and understand the process.

**Input:**
.. code-block:: text

   Multiply the matrices: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]

**Expected Solution Process:**
1. Check dimension compatibility
2. Apply matrix multiplication rules
3. Show element-by-element calculation
4. Present the result matrix

Matrix Decompositions
---------------------

LU Decomposition
~~~~~~~~~~~~~~~~

**Problem:**
Decompose a matrix into lower and upper triangular matrices.

**Input:**
.. code-block:: text

   Perform LU decomposition on: [[4, 3], [6, 3]]

**Expected Solution Process:**
1. Find permutation matrix P (if needed)
2. Decompose into L (lower triangular) and U (upper triangular)
3. Verify that P × A = L × U
4. Display all three matrices with explanations

**Why It's Useful:**
- Efficient solving of linear systems
- Matrix determinant calculation
- Foundation for other numerical methods

**Advanced Example:**
.. code-block:: text

   Find the LU decomposition of: [[2, 1, 1], [4, 3, 3], [8, 7, 9]]

QR Decomposition
~~~~~~~~~~~~~~~~

**Problem:**
Decompose a matrix into orthogonal and upper triangular matrices.

**Input:**
.. code-block:: text

   Compute QR decomposition of: [[1, 1], [1, 0], [0, 1]]

**Expected Solution Process:**
1. Apply Gram-Schmidt process or Householder reflections
2. Construct orthogonal matrix Q
3. Construct upper triangular matrix R  
4. Verify that A = Q × R
5. Check orthogonality: Q^T × Q = I

**Applications You'll Learn About:**
- Least squares solutions
- Eigenvalue algorithms
- Orthogonal basis construction

Singular Value Decomposition (SVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Decompose a matrix into its singular values and vectors.

**Input:**
.. code-block:: text

   Perform SVD on the matrix: [[3, 2, 2], [2, 3, -2]]

**Expected Solution Process:**
1. Compute A^T × A and A × A^T
2. Find eigenvalues and eigenvectors
3. Construct U, Σ, and V^T matrices
4. Verify that A = U × Σ × V^T
5. Display singular values in descending order

**What Makes SVD Special:**
- Works for any matrix (not just square)
- Reveals matrix rank and null space
- Foundation for Principal Component Analysis (PCA)
- Used in data compression and noise reduction

Eigenvalue Problems
-------------------

Finding Eigenvalues and Eigenvectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Find eigenvalues and eigenvectors of a symmetric matrix.

**Input:**
.. code-block:: text

   Find the eigenvalues and eigenvectors of: [[3, 1], [1, 3]]

**Expected Solution Process:**
1. Form characteristic equation: det(A - λI) = 0
2. Solve polynomial equation for eigenvalues
3. For each eigenvalue, solve (A - λI)x = 0 for eigenvectors
4. Normalize eigenvectors
5. Verify: A × v = λ × v

**Real-World Example:**
.. code-block:: text

   A system has the matrix [[2, -1], [-1, 2]]. Find its natural frequencies (eigenvalues).

**What You'll Understand:**
- Physical meaning of eigenvalues and eigenvectors
- Characteristic polynomial method
- Geometric vs. algebraic multiplicity
- Diagonalization conditions

Matrix Diagonalization
~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Diagonalize a matrix using its eigenvalues and eigenvectors.

**Input:**
.. code-block:: text

   Diagonalize the matrix: [[1, 2], [2, 1]]

**Expected Solution Process:**
1. Find eigenvalues and eigenvectors
2. Construct matrix P from eigenvectors
3. Form diagonal matrix D from eigenvalues
4. Verify that A = P × D × P⁻¹
5. Show the diagonalization

**Complex Eigenvalues Example:**
.. code-block:: text

   Find eigenvalues of the rotation matrix: [[0, -1], [1, 0]]

Linear Systems
--------------

Solving Ax = b
~~~~~~~~~~~~~~

**Problem:**
Solve a system of linear equations.

**Input:**
.. code-block:: text

   Solve the system: 2x + y = 5, x + 3y = 8

**Alternative Matrix Form:**
.. code-block:: text

   Solve Ax = b where A = [[2, 1], [1, 3]] and b = [5, 8]

**Expected Solution Process:**
1. Set up the augmented matrix
2. Apply Gaussian elimination or use matrix inverse
3. Express solution in exact and decimal form
4. Verify solution by substitution

Overdetermined Systems
~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Find the least squares solution when there are more equations than unknowns.

**Input:**
.. code-block:: text

   Find the least squares solution for: x + y = 1, x + 2y = 2, 2x + y = 2

**Expected Solution Process:**
1. Set up A^T × A × x = A^T × b
2. Solve the normal equations
3. Calculate residual and fitting error
4. Explain why this is the "best" solution

Advanced Applications
---------------------

Principal Component Analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Perform dimensionality reduction using PCA.

**Input:**
.. code-block:: text

   Perform PCA on the data matrix: [[1, 2], [3, 4], [5, 6], [7, 8]]

**Expected Solution Process:**
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors of covariance matrix
4. Sort by eigenvalue magnitude
5. Select principal components
6. Transform data to new coordinate system

Matrix Powers and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Compute high powers of a matrix efficiently.

**Input:**
.. code-block:: text

   Calculate A^10 where A = [[1, 1], [1, 0]] (Fibonacci matrix)

**Expected Solution Process:**
1. Diagonalize the matrix: A = P × D × P⁻¹
2. Use the property: A^n = P × D^n × P⁻¹  
3. Compute D^n (diagonal elements raised to power n)
4. Reconstruct A^n

**Matrix Exponential Example:**
.. code-block:: text

   Compute the matrix exponential e^A for A = [[0, 1], [-1, 0]]

Numerical Considerations
------------------------

Condition Numbers
~~~~~~~~~~~~~~~~~

**Problem:**
Assess the numerical stability of matrix operations.

**Input:**
.. code-block:: text

   Calculate the condition number of: [[1, 1], [1, 1.0001]]

**What You'll Learn:**
- How small changes in input affect output
- When matrix operations become unreliable
- Relationship between condition number and numerical precision

**Well-Conditioned Example:**
.. code-block:: text

   Compare condition numbers: [[2, 0], [0, 3]] vs [[1, 0.999], [0.999, 1]]

Matrix Norms
~~~~~~~~~~~~

**Problem:**
Calculate different matrix norms and understand their meanings.

**Input:**
.. code-block:: text

   Calculate the Frobenius norm and spectral norm of: [[3, 4], [0, 5]]

**Expected Solution Process:**
1. Frobenius norm: √(sum of squares of all elements)
2. Spectral norm: largest singular value
3. Infinity norm: maximum row sum
4. 1-norm: maximum column sum

Visualization Features
----------------------

Understanding Matrix Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you input a matrix problem, the system automatically generates visualizations:

**Matrix Heatmap:**
- Color intensity represents element values
- Patterns reveal matrix structure
- Useful for spotting symmetry, sparsity, or special structure

**Eigenvalue Plots:**
- Complex plane visualization for eigenvalues
- Real vs. imaginary parts
- Magnitude and phase information

**Decomposition Displays:**
- Side-by-side comparison of original and decomposed matrices
- Factor matrices shown separately
- Reconstruction verification

Interactive Examples
--------------------

Try These Step-by-Step
~~~~~~~~~~~~~~~~~~~~~~

**Beginner Level:**

1. **Simple 2×2 determinant:**
   ``Calculate determinant of [[a, b], [c, d]] with a=1, b=2, c=3, d=4``

2. **Identity matrix verification:**
   ``Show that [[1, 0], [0, 1]] is the identity matrix by multiplying with [[2, 3], [4, 5]]``

**Intermediate Level:**

3. **Eigenvalue problem:**
   ``Find eigenvalues of the covariance matrix [[4, 2], [2, 3]]``

4. **Linear system with unique solution:**
   ``Solve: 3x + 2y = 7, x - y = 1``

**Advanced Level:**

5. **SVD application:**
   ``Use SVD to find the rank of [[1, 2, 3], [2, 4, 6], [1, 2, 4]]``

6. **Matrix exponential:**
   ``Compute e^(At) where A = [[0, 1], [-1, 0]] and t = π/2``

Common Pitfalls and How to Avoid Them
-------------------------------------

**Singular Matrices:**
- Always check determinant before computing inverse
- Understand when solutions don't exist or aren't unique

**Numerical Precision:**
- Be aware of floating-point limitations
- Use exact symbolic computation when possible

**Dimension Mismatches:**
- Verify matrix dimensions before operations
- Understand when operations are undefined

**Interpretation:**
- Eigenvalues have units and physical meaning
- Eigenvectors show direction, not just numbers

Best Practices
--------------

**Problem Setup:**
- Clearly define your matrices using standard notation
- Specify whether you want exact or numerical results
- Include units and context when relevant

**Verification:**
- Always check that your solution satisfies the original equation
- Use the verification features provided by the system
- Cross-check with alternative methods when possible

**Learning:**
- Study the step-by-step solutions to understand methods
- Try variations of problems to build intuition
- Connect linear algebra concepts to real applications

Ready for More?
---------------

After mastering these linear algebra concepts, explore:

* **Optimization Tutorial**: See how eigenvalues appear in optimization
* **Statistics Tutorial**: Learn about covariance matrices and PCA
* **Advanced Features**: Sparse matrices, iterative methods, matrix functions

Linear algebra is the foundation of much of modern mathematics and data science. The Math AI Agent helps you build this foundation with accurate computations, clear explanations, and helpful visualizations.

**Happy computing!** 🔢📊