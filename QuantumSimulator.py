#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on May 4, 2021

@author: Matthew
'''

from math import log2
from random import randrange, uniform


################ START EXAMPLE CODE DEMONSTRATION ################


if __name__ == '__main__':
    '''The driver code for testing the Qubit and Matrix classes'''
    
    # Testing if H|+> = |0>
    print("Testing if H|+> = |0>:")
    
    reg1 = Qubit(states=[1, 1])
    
    print(f"reg1 = |+> =\n{reg1.peek()}")
    
    reg1.applyGate(Qubit.H)
    
    print(f"reg1 = H|+> = |0> =\n{reg1.peek()}")
    
    
    # Testing if Z|-> = |+>
    print("Testing if Z|-> = |+>:")
    
    reg2 = Qubit(states=[1, -1])
    
    print(f"reg2 = |-> =\n{reg2.peek()}")
    
    reg2.applyGate(Qubit.Z)
    
    print(f"reg2 = Z|-> = |+> =\n{reg2.peek()}")
    
    
    # Testing entanglement between qubit registers 1 and 2
    print("Entangling qubit registers 1 and 2:")
    
    reg2.entangle(reg1)
    
    print(f"reg2 = reg1 = |-> ⊗ |+> = |-+> =\n{reg2.peek()}")
    
    try:
        print(f"reg1 =\n{reg1.peek()}")
        
    except AttributeError:
        print("reg1 no longer exists in the code, "
              + "as its qubit is entangled with that of reg2\n")
    
    # Testing entangled qubit measurement
    print("Testing entangled qubit measurement:")
    
    for _ in range(3):
        print(f"reg2 measured =\n{reg2}")
    
# End of __main__


# Possible output A:

# Testing if H|+> = |0>:
# reg1 = |+> =
# [0.7071067811865476]
# [0.7071067811865476]
#
# reg1 = H|+> = |0> =
# [1.0000000000000002]
# [0.0]
#
# Testing if Z|-> = |+>:
# reg2 = |-> =
# [0.7071067811865476]
# [-0.7071067811865476]
#
# reg2 = Z|-> = |+> =
# [0.7071067811865476]
# [0.7071067811865476]
#
# Entangling qubit registers 1 and 2:
# reg2 = reg1 = |-> ⊗ |+> = |-+> =
# [0.7071067811865477]
# [0.7071067811865477]
# [0.0]
# [0.0]
#
# reg1 no longer exists in the code,
# as its qubit is entangled with that of reg2
#
# Testing entangled qubit measurement:
# reg2 measured =
# [1]
# [0]
# [0]
# [0]
#
# reg2 measured =
# [1]
# [0]
# [0]
# [0]
#
# reg2 measured =
# [1]
# [0]
# [0]
# [0]


# Possible output B (excluding same beginning portion):

# Testing entangled qubit measurement:
# reg2 measured =
# [0]
# [1]
# [0]
# [0]
#
# reg2 measured =
# [0]
# [1]
# [0]
# [0]
#
# reg2 measured =
# [0]
# [1]
# [0]
# [0]


################# END EXAMPLE CODE DEMONSTRATION #################


class Matrix():
    '''A class describing standard mathematical matrices
    
    Attributes:
        - self.mat    A list of lists containing the values
    
    '''
    
    def __init__(self, cols=2, rows=2, data=None):
        '''The matrix initialization method/constructor
        
        Params:
            - cols    The quantity of columns in the matrix
            - rows    The quantity of rows in the matrix
            - data    A flat list containing at least as
                      many elements as the matrix demands
        
        '''
        
        if data == None:
            
            # Initializing zero matrix
            self.mat = [
                [0 for j in range(cols)]
                for i in range(rows)
            ]
            
        else:
            
            # Initializing matrix
            self.mat = [
                [data[cols * i + j] for j in range(cols)]
                for i in range(rows)
            ]
        
    # End of __init__
    
    class IllegalDimensionsException(Exception):
        ''' An Exception used when the dimensions
        of a matrix fail to meet certain criteria'''
        
        pass
    
    # End of IllegalDimensionsException
    
    def rows(self):
        '''Returns the quantity of rows in the matrix'''
        
        return len(self.mat)
        
    # End of rows
    
    def cols(self):
        '''Returns the quantity of columns in the matrix'''
        
        return len(self.mat[0])
        
    # End of rows
    
    def dotProd(self, matrix):
        '''Returns the matrices' dot product
        
        Params:
            - matrix    the matrix being multiplied
        
        '''
        
        # Checking dimensions
        if self.cols() != matrix.rows():
            raise Matrix.IllegalDimensionsException(
                "Quantity of columns in self doesn't match"
                + "quantity of rows in secondary matrix")
        
        # Multiplying matrices
        return Matrix(
            cols=matrix.cols(),
            rows=self.rows(),
            data=[
                sum(
                    self.get(i, k) * matrix.get(k, j)
                    for k in range(matrix.rows())
                )
                
                for j in range(matrix.cols())
                for i in range(self.rows())
            ]
        )
        
    # End of prod
    
    def scalarProd(self, scalar:float):
        '''Returns a matrix scaled by the scalar *scalar*
        
        Params:
            - scalar    The factor of which to scale the matrix by
        
        '''
        
        return Matrix(
            cols=self.cols(),
            rows=self.rows(),
            data=[e * scalar for row in self.mat for e in row]
        )
        
    # End of scalarProd
    
    def get(self, i:int, j:int):
        '''Returns the indexed element
        
        Params:
            - i    The row index
            - j    The column index
        
        '''
        
        return self.mat[i][j]
    
    # End of get
    
    def set(self, i:int, j:int, e):
        '''Sets the indexed element
        
        Params:
            - i    The row index
            - j    The column index
            - e    The element to set
        
        '''
        
        self.mat[i][j] = e
    
    # End of set
    
    def transpose(self):
        '''Returns the matrix transpose'''
        
        out = Matrix(cols=self.rows(), rows=self.cols())
        
        # Transferring elements
        for i in range(self.cols()):
            for j in range(self.rows()):
                out.mat[i][j] = self.mat[j][i]
        
        return out
        
    # End of transpose()
    
    def tensorProd(self, matrix):
        '''Returns the tensor/Kronecker product of the matrices
        
        Params:
            - matrix    The second matrix in the product
        
        '''
        
        return Matrix(
            cols=self.cols() * matrix.cols(),
            rows=self.rows() * matrix.rows(),
            data=[
                self.mat[i][j] * matrix.mat[p][q]
                    for p in range(matrix.rows())
                        for q in range(matrix.cols())
                            for i in range(self.rows())
                                for j in range(self.cols())
            ]
        )
        
    # End of tensorProd(matrix)
    
    def blockDiag(self, matrix):
        '''
        Generates a diagonal block matrix from the given matrices, i.e.:
        
        ⎧ self    0  ⎫
        ⎩ 0   matrix ⎭
        
        Params:
            - matrix    The second matrix in the block
        
        '''
        
        out = Matrix(cols=self.cols() + matrix.cols(), rows=self.rows() + matrix.rows())
        
        # Adding top-left submatrix
        for i in range(self.rows()):
            for j in range(self.cols()):
                out.mat[i][j] = self.mat[i][j]
        
        # Adding bottom-right submatrix
        for i in range(matrix.rows()):
            for j in range(matrix.cols()):
                out.mat[self.rows() + i][self.cols() + j] = matrix.mat[i][j]
        
        return out
        
    # End of blockDiag(matrix)
    
    def normal(self):
        ''' Returns the matrix normal'''
        
        normal = 0
        
        # Summing overall state probabilities
        for row in self.mat:
            for e in row:
                normal += abs(e) ** 2
        
        return normal ** -0.5
        
    # End of normal()
    
    def normalize(self):
        '''Returns the matrix normalized'''
        
        normal = self.normal()
        
        # Checking if vector normalization is possible
        if normal == 0:
            raise Matrix.IllegalDimensionsException("Matrix cannot be normalized")
        
        return self.scalarProd(normal)
        
    # End of normalize()
    
    def __add__(self, matrix):
        '''Returns the sum of the matrices
        
        Params:
            - matrix    The matrix being added
        
        '''
        
        # Checking if dimensions are equal
        if self.cols() != matrix.cols() or self.rows() != matrix.rows():
            raise Matrix.IllegalDimensionsException()
        
        return Matrix(
            cols=self.cols(),
            rows=self.rows(),
            data=[
                self.get(i, j) + matrix.get(i, j)
                for i in range(self.rows())
                for j in range(self.cols())
            ])
        
    # End of __add__()
    
    def __sub__(self, matrix):
        '''Returns the difference of the matrices
        
        Params:
            - matrix    The matrix being subtracted by (i.e, returns self - matrix)
        
        '''
        
        # Checking if dimensions are equal
        if self.cols() != matrix.cols() or self.rows() != matrix.rows():
            raise Matrix.IllegalDimensionsException()
        
        return Matrix(
            cols=self.cols(),
            rows=self.rows(),
            data=[
                self.get(i, j) - matrix.get(i, j)
                for i in range(self.rows())
                for j in range(self.cols())
            ])
        
    # End of __add__()
    
    def __str__(self):
        '''Returns the matrix as a string'''
        
        out = ''
        
        for r in self.mat:
            out += '[' + str(r) + ']\n'
        
        return out[:-1]
        
    # End of __str__()
    
    def __repr__(self):
        '''Returns a constructor representation of the object'''
        
        return f"Matrix(data={[e for row in self.mat for e in row]},"
        + f"cols={self.cols()}, rows={self.cols()})"
        
    # End of __repr__()
    
# End of Matrix


class Qubit():
    ''' A class representing in statevector form either
    a single quantum bit (qubit) or its entangled qubits 
    
    Attributes:
        - self.__vec    The internal qubit statevector
    
    '''
    
    # Statevector representations in ket notation:
    
    # |0> represents  ⎧ 1 ⎫
    #                 ⎩ 0 ⎭
                   
    # |1> represents  ⎧ 0 ⎫
    #                 ⎩ 1 ⎭
                   
    # |+> represents sqrt(1/2)  *  ⎧ 1 ⎫
    #                              ⎩ 1 ⎭
                               
    # |-> represents sqrt(1/2)  *  ⎧  1 ⎫
    #                              ⎩ -1 ⎭
    
    def __init__(self, states=None, qubits=1, value=0):
        '''The qubit initialization method/constructor,
        which defaults to |0>
        
        Params:
            - states   The statevector states
            - qubits   The quantity of qubits entangled
            - value    The initial state, represented
                       in ket notation (decimal) by | *value* >
        '''
        
        if qubits < 1 or value < 0:
            raise Matrix.IllegalDimensionsException(
                "Qubit cannot have less than two states")
            
        elif states == None:
            
            # Initializing statevector with one possible outcome
            self.__vec = Matrix(cols=1, rows=2 ** qubits)
            self.__setState(value, 1)
            
        elif log2(len(states)) % 1 == 0:
            
            # Creating normalized statevector from input states
            self.__vec = Matrix(
                cols=1,
                rows=len(states),
                data=states
            ).normalize()
            
        else:
            raise Matrix.IllegalDimensionsException(
                "Quantity of qubit states are not a natural power of two")
        
    # End of __init__
    
    
    # H is the Hadamard gate, which maps
    # |0> to |+>, |1> to |-> and vise versa:
    #
    # sqrt(1/2)  *  ⎧ 1,  1 ⎫
    #               ⎩ 1, -1 ⎭
    
    H = Matrix(data=[1, 1, 1, -1]).scalarProd(0.5 ** 0.5)
    
    
    # X is a 2x2 matrix with ones along the anti-diagonal
    # known as the Pauli-X gate, or otherwise known as
    # the NOT or X gate:
    #
    #  ⎧ 0, 1 ⎫
    #  ⎩ 1, 0 ⎭
    
    X = Matrix(data=[0, 1, 1, 0])
    
    
    # Y is the Pauli-Y or Y gate:
    #
    #  ⎧ 0  -i ⎫
    #  ⎩ i   0 ⎭
    
    Y = Matrix(data=[0, -1j, 1j, 0])
    
    
    # Z is known as either the Pauli-Z, Z or phase-flip gate:
    #
    #  ⎧ 1   0 ⎫
    #  ⎩ 0  -1 ⎭
    
    Z = Matrix(data=[1, 0, 0, -1])
    
    
    # SQRTNOT is the sqrt(X) or sqrt(NOT) gate
    # and maps |0> to [1+i] and |1> to [1-i]:
    #                 [1-i]            [1+i]
    #
    # 1/2  *  ⎧ 1+i  1-i ⎫
    #         ⎩ 1-i  1+i ⎭
    
    SQRTNOT = Matrix(data=[1+1j, 1-1j, 1-1j, 1+1j]).scalarProd(0.5)
    
    
    # I2 is the 2x2 identity matrix known simply as the identity,
    # 2x2 identity or I gate:
    #
    #  ⎧ 1  0 ⎫
    #  ⎩ 0  1 ⎭
    
    I2 = Matrix(data=[1, 0, 0, 1])
    
    
    # I4 is the 4x4 identity matrix:
    #
    #    1  0  0  0 
    #  ⎛ 0  1  0  0 ⎞
    #    0  0  1  0
    #  ⎝ 0  0  0  1 ⎠
    
    I4 = I2.blockDiag(I2)
    
    
    # CNOT is the controlled NOT or controlled X gate:
    #
    #    1  0  0  0 
    #  ⎛ 0  1  0  0 ⎞
    #    0  0  0  1 
    #  ⎝ 0  0  1  0 ⎠
    
    CNOT = I2.blockDiag(X)
    
    
    # CCNOT is the controlled controlled NOT or Toffoli gate:
    #
    #    1  0  0  0  0  0  0  0 
    #  ⎛ 0  1  0  0  0  0  0  0 ⎞
    #  ⎜ 0  0  1  0  0  0  0  0 ⎟
    #  ⎜ 0  0  0  1  0  0  0  0 ⎟
    #  ⎜ 0  0  0  0  1  0  0  0 ⎟
    #  ⎜ 0  0  0  0  0  1  0  0 ⎟
    #    0  0  0  0  0  0  0  1 
    #  ⎝ 0  0  0  0  0  0  1  0 ⎠
    
    CCNOT = I4.blockDiag(CNOT)
    
    
    # SWAP is the self-explanatory SWAP gate:
    #
    #    1  0  0  0 
    #  ⎛ 0  0  1  0 ⎞
    #    0  1  0  0 
    #  ⎝ 0  0  0  1 ⎠
    
    SWAP = Matrix(cols=4, rows=4, data=[1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1])
    
    
    # CSWAP is obviously also the controlled SWAP gate:
    #
    #    1  0  0  0  0  0  0  0 
    #  ⎛ 0  1  0  0  0  0  0  0 ⎞
    #  ⎜ 0  0  1  0  0  0  0  0 ⎟
    #  ⎜ 0  0  0  1  0  0  0  0 ⎟
    #  ⎜ 0  0  0  0  1  0  0  0 ⎟
    #  ⎜ 0  0  0  0  0  0  1  0 ⎟
    #    0  0  0  0  0  1  0  0 
    #  ⎝ 0  0  0  0  0  0  0  1 ⎠
    
    CSWAP = I4.blockDiag(SWAP)
    
    
    def __get(self, i:int):
        '''Returns the indexed element in the statevector
        
        Params:
            - i    The index
        
        '''
        
        return self.__vec.get(i, 0)
        
    # End of __get(i)
    
    def __setState(self, i:int, s:float):
        '''Sets the indexed element in the statevector
        
        Params:
            - i    The index
            - s    The state
        
        '''
        
        self.__vec.set(i, 0, s)
        
    # End of __setState(i)
    
    def entangle(self, qubit):
        '''Entangles qubits through vector tensor product,
        stores entangled qubit in *self* and deletes
        the vectorstate attribute *vec* of *qubit*,
        thus leaving *qubit* open for reinitialization
        
        Params:
            - qubit    The qubit to be entangled and its
                       statevector attribute deleted
        
        '''
        
        self.__vec = self.__vec.tensorProd(qubit.__vec)
        del qubit.__vec
        
    # End of tensor
    
    def applyGate(self, gate):
        '''Applies the quantum logic gate (unitary matrix)
        *gate* to the qubit through the dot product.
        It is not checked whether the gate is unitary,
        as it would often waste unnecessary resources
        on the common matrices defined: the user is
        expected to use unitary matrices
        
        Params:
            - gate    The unitary matrix acting
                      as a quantum logic gate
        
        '''
        
        self.__vec = gate.dotProd(self.__vec)
        
    # End of applyGate
    
    def measure(self):
        '''Measures qubit and collapses qubit
        into a probabilistically chosen state'''
        
        def __collapse(p):
            '''Collapses qubit to state *p*
            
            Params:
            - p    The state to collapse the qubit into
        
            '''
            
            # Choosing statevector index of equal probabilities
            indices = [i for i in range(self.__vec.rows()) if self.__get(i) == p]
            index = randrange(len(indices))
            
            # Collapsing qubit
            self.__vec = Matrix(cols=1, rows=self.__vec.rows())
            self.__setState(indices[index], 1)
            
        # End of __collapse(i)
        
        # Getting probabilities
        prob = [e[0] ** 2 for e in self.__vec.mat if e[0] != 0]
        
        # Sorting probabilities in descending order
        prob.sort(reverse=True)
        
        # Probabilistically finding state to collapse qubit into
        for i in range(len(prob)):
            if uniform(0, 1) <= prob[i]:
                __collapse(prob[i] ** 0.5)
                
                return None
        
        # Collapsing qubit to least likely state
        __collapse(prob[-1] ** 0.5)
        
    # End of measure
    
    def peek(self):
        '''Returns qubit state without measuring it,
        which violates quantum physics.
        Intended for classical statevector testing only.'''
        
        return str(self.__vec)
        
    # End of peek
    
    def clone(self):
        '''Returns a deep copy of the qubit without measuring it,
        which violates quantum physics.
        Intended for shorter, simpler code only.'''
        
        out = Qubit(states=[s[0] for s in self.__vec])
        
        return out
        
    # End of clone
    
    def __str__(self):
        '''Measures the statevector and returns it in string format'''
        
        self.measure()
        
        return str(self.__vec)
    
    # End of __str__
    
    def __repr__(self):
        '''Returns a constructor representation of the object'''
        
        return f'Qubit(states={[s[0] for s in self.__vec]})'
    
    # End of __repr__
    
    def __len__(self):
        '''Returns the quantity of qubits entangled in this entity'''
        
        return log2(len(self.__vec))
        
    # End of __len__
        
# End of Qubit
