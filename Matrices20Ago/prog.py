from timeit import timeit
import numpy as np
import pytest

"""
Actividad 1.1 Implementación de la técnica de programación "divide y vencerás"
Nombre: Angel Orlando Vazquez Morales
Matricula: A01659000
Análisis y diseño de algoritmos avanzados (Gpo 652)
"""

class Program:  
    def bruteMethod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Método para multiplicar matrices a fuerza bruta (de la forma AB = C)
        Complejidad de tiempo: O(n^3)

        Args:
            a (np.ndarray): Primera matriz
            b (np.ndarray): Segunda matriz
        Returns:
            np.ndarray: Resultado de la multiplicación de matrices
        """
        rows: int = len(a)
        cols: int = len(b[0])
        result: np.ndarray = np.zeros([rows, cols], dtype=np.bool)

        for i in range(rows):
            for j in range(cols):
                result[i, j] = 0
                for k in range(rows):
                    result[i, j] = result[i, j] | (a[i, k] & b[k, j])
        return result


    def strassenMethod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: 
        """Método para multiplicar matrices a fuerza bruta (de la forma AB = C)
        Este método solo puede multiplicar matrices cuadradas del mismo tamaño 1x1, 2x2 o de la forma x = 2^n
        Complejidad de tiempo: O(n^log2(7))

        Args:
            a (np.ndarray): Primera matriz cuadrada nxn
            b (np.ndarray): Segunda matriz cuadrada nxn
        Returns:
            np.ndarray: Resultado de la multiplicación de matrices
        """
        a: np.ndarray = np.astype(a, np.int32)
        b: np.ndarray = np.astype(b, np.int32)

        size: int = len(b)
        result: np.ndarray = np.zeros([size, size], dtype=np.int32)
 
        if size <= 1:
            result[0][0] = a[0][0] + b[0][0]
        elif size <= 2:
            m1 = ( a[0, 0] + a[1, 1] ) * ( b[0, 0] + b[1, 1] )
            m2 = ( a[1, 0] + a[1, 1] ) * b[0, 0]
            m3 = a[0, 0] * ( b[0, 1] - b[1, 1] )
            m4 = a[1, 1] * ( b[1, 0] - b[0, 0] )
            m5 = ( a[0, 0] + a[0, 1] ) * b[1, 1]
            m6 = ( a[1, 0] - a[0, 0] ) * ( b[0, 0] + b[0, 1] )
            m7 = ( a[0, 1] - a[1, 1] ) * ( b[1, 0] + b[1, 1] )
        else:
            a1, b1, c1, d1 = self.subdivide(a)
            a2, b2, c2, d2 = self.subdivide(b)
            m1 = self.strassenMethod(a1 + d1, a2 + d2).astype(np.int32)
            m2 = self.strassenMethod(c1 + d1, a2).astype(np.int32)
            m3 = self.strassenMethod(a1, b2 - d2).astype(np.int32)
            m4 = self.strassenMethod(d1, c2 - a2).astype(np.int32)
            m5 = self.strassenMethod(a1 + b1, d2).astype(np.int32)
            m6 = self.strassenMethod(c1 - a1, a2 + b2).astype(np.int32)
            m7 = self.strassenMethod(b1 - d1, c2 + d2).astype(np.int32)
        result[:size//2, :size//2] = m1 + m4 - m5 + m7
        result[:size//2, size//2:] = m3 + m5
        result[size//2:, :size//2] = m2 + m4
        result[size//2:, size//2:] = m1 - m2 + m3 + m6
        
        result = result.astype(np.bool)
        return result


    def booleanMethod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Método para multiplicar matrices a fuerza bruta (de la forma AB = C)
        Complejidad de tiempo: O(n^2) en el mejor caso, O(n^3) en el peor caso.

        Args:
            a (np.ndarray): Primera Matriz
            b (np.ndarray): Segunda Matriz
        Returns:
            np.ndarray: Resultado de la multiplicación de matrices
        """
        size: int = len(a) # Cantidad de filas y columnas (NxN)
        na: int = a.sum() # Cantidad de 1's en la primera matriz
        nb: int = b.sum() # Cantidad de 1's en la segunda matriz
        result: np.ndarray = np.zeros([size, size], dtype=np.bool)

        # En el caso que NA tenga menos 1's que NB, entonces se hace el calculo de B^T x A^T = C^T
        if na < nb:
            c = a.transpose().copy()
            a = b.transpose().copy()
            b = c

        # Recorrido por todas las filas de la primera matriz.
        for row in range(size):
            # Lista de 1's en la fila actual
            ones_positions: list[int] = [i for i in range(size) if a[row, i] == 1]
            # Recorrido de columnas de la matriz B
            for j in range(size):
                for k in ones_positions:
                    if b[k, j] == 1:
                        result[row, j] = 1
                        break
        
        # Para este caso, hay que transponer C^T para volver a obtener C
        if na < nb:
            result = result.transpose()
        return result
    

    def subdivide(self, matrix: np.ndarray) -> tuple[np.ndarray]:
        """Método que divide matrices cuadradas de longitud par en cuatro cuadrantes.

        Args:
            matrix (np.ndarray): Matriz cuadrada a dividir
        Returns:
            tuple[np.ndarray]: Tupla de cuatro matrices cuadradas
        """
        n: int = len(matrix)
        a: np.ndarray = matrix[:n//2, :n//2].copy()
        b: np.ndarray = matrix[:n//2, n//2:].copy()
        c: np.ndarray = matrix[n//2:, :n//2].copy()
        d: np.ndarray = matrix[n//2:, n//2:].copy()
        return (a, b, c, d)


def testBruteMethod():
    """Método que realiza pruebas unitarias al mètodo de fuerza bruta
    Complejidad: O(n^3)
    """
    prog = Program()
    A: np.ndarray = np.array([ [1, 0], [0, 1] ], dtype=np.bool)
    B: np.ndarray = np.array([ [0, 1], [1, 1] ], dtype=np.bool)

    C: np.ndarray = np.array([ [0] ], dtype=np.bool)
    D: np.ndarray = np.array([ [1] ], dtype=np.bool)

    E: np.ndarray = np.array([ [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.bool)
    F: np.ndarray = np.array([ [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.bool)

    C1: np.ndarray = prog.bruteMethod(A, B) 
    C2: np.ndarray = prog.bruteMethod(C, D)
    C3: np.ndarray = prog.bruteMethod(E, F)

    assert (C1 == np.array([ [0, 1], [1, 1] ], dtype=np.bool)).all()
    assert (C2 == np.array( [ [0] ], dtype=np.bool)).all()
    assert (C3 == np.array( [[77, 136, 74, 96], [36, 99, 46, 86], [54, 114, 58, 89], [35, 41, 42, 12]], dtype=np.bool)).all()


def testStraissenMethod():
    """Método que realiza pruebas unitarias al mètodo de straissen
    Complejidad: O(n^log2(7))
    """
    prog = Program()
    A: np.ndarray = np.array([ [1, 0], [0, 1] ], dtype=np.bool)
    B: np.ndarray = np.array([ [0, 1], [1, 1] ], dtype=np.bool)

    C: np.ndarray = np.array([ [0] ], dtype=np.bool)
    D: np.ndarray = np.array([ [1] ], dtype=np.bool)

    E: np.ndarray = np.array([ [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.bool)
    F: np.ndarray = np.array([ [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.bool)

    C1: np.ndarray = prog.strassenMethod(A, B) 
    C2: np.ndarray = prog.strassenMethod(C, D)
    C3: np.ndarray = prog.strassenMethod(E, F)

    assert (C1 == np.array([ [0, 1], [1, 1] ], dtype=np.bool)).all()
    assert (C2 == np.array( [ [0] ], dtype=np.bool)).all()
    assert (C3 == np.array( [[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]], dtype=np.bool)).all()


def testBooleanMethod():
    """Mètodo que realiza pruebas unitarias al método booleano
    Complejidad promedio: O(n^2), Complejidad en el peor caso: O(n^3)
    """
    prog = Program()
    A: np.ndarray = np.array([ [0, 1, 0], [1, 0, 1], [0, 1, 1] ], dtype=np.bool)
    B: np.ndarray = np.array([ [1, 1, 0], [0, 0, 0], [0, 1, 1] ], dtype=np.bool) 
    
    C: np.ndarray = np.array([ [1] ], dtype=np.bool)
    D: np.ndarray = np.array([ [1] ], dtype=np.bool)

    E: np.ndarray = np.array([ [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.bool)
    F: np.ndarray = np.array([ [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.bool)

    C1: np.ndarray = prog.booleanMethod(A, B) # AB = C1
    C2: np.ndarray = prog.booleanMethod(B, A) # BA = C2
    C3: np.ndarray = prog.booleanMethod(C, D) # CD = C3
    C4: np.ndarray = prog.booleanMethod(E, F) # EF = C4

    assert (C1 == np.array([ [0, 0, 0], [1, 1, 1], [0, 1, 1] ], dtype=np.bool)).all()
    assert (C2 == np.array([ [1, 1, 1], [0, 0, 0], [1, 1, 1] ], dtype=np.bool)).all()
    assert (C3 == np.array([ [1] ], dtype=np.bool)).all()
    assert (C4 == 
            np.array([ [0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]], 
                     dtype=np.bool)).all()



if __name__ == "__main__":
    def ejemplo():
        E: np.ndarray = np.array([ [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.int16)
        F: np.ndarray = np.array([ [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.int16)
        prog: Program = Program()

        D1: np.ndarray = prog.strassenMethod(E, F)
        D2: np.ndarray = prog.booleanMethod(E, F)
        # Ejemplo debe dar [[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]
        print(D1)
        print(D2)


    """ Para ejecutar los tiempos de velocidad: python prog.py """
    """ Para ejecutar los test unitarios con pytest: pytest -v prog.py """
    A2: np.ndarray = np.array([ [0, 1, 0, 1, 0, 0, 1, 1], 
                                [1, 0, 1, 1, 0, 1, 1, 1], 
                                [0, 1, 1, 1, 1, 0, 0, 1],
                                [1, 1, 0, 0, 0, 0, 0, 1], 
                                [0, 0, 0, 0, 1, 0, 0, 1],
                                [1, 1, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 1] ])
    B2: np.ndarray = np.array([ [0, 1, 1, 1, 0, 0, 0, 0], 
                                [1, 0, 1, 1, 1, 1, 0, 0], 
                                [0, 1, 0, 0, 1, 0, 1, 1],
                                [1, 1, 0, 1, 1, 0, 0, 0], 
                                [0, 0, 1, 1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 1],
                                [1, 1, 0, 0, 0, 0, 1, 1] ])

    prog = Program()
    
    naiveTime: float = timeit(lambda: prog.bruteMethod(A2, B2), number=10000)
    strassenTime: float = timeit(lambda: prog.strassenMethod(A2, B2), number=10000)
    booleanTime: float = timeit(lambda: prog.booleanMethod(A2, B2), number=10000)
    print(f"tiempo de ejecución de Brute-Force: {naiveTime: .3f}s")
    print(f"tiempo de ejecución de Strassen: {strassenTime: .3f}s")
    print(f"tiempo de ejecución de Boolean: {booleanTime: .3f}s")

    print()
    ejemplo()