import pytest
"""
Actividad 1.2 Implementación de la técnica de programación "Programación dinámica" 
Nombres y matrículas:
- Angel Orlando Vazquez Morales - A01659000
- Sergio Morales Gónzales - A01694200 (cambiar esto XD)
Análisis y diseño de algoritmos avanzados (Gpo 652)
"""
class WeddingShopping:
    """
    Una clase que representa la solución al problema de Wedding Shopping.

    ...

    Attributes
    ----------
    m : int
        representa la cantidad de dinero del problema Wedding Shopping (1 <= m <= 200)
    c : int
        representa la cantidad de categorías de ropa a elegir (1 <= c <= 20)
    garnments : list[list[int]]
        representa una lista de listas de precios por categoria

    Methods
    -------
    weddingTopDown(money: int, c: int)
        Método que resuelve el problema de Wedding Shopping de forma Top-Down (Recursiva y con Memoisación).

    weddingBottomUp(money: int, c: int)
        Método que resuelve el problema de Wedding Shopping de forma Bottom Up (Iterativamente y con Tabulación)
    """
    memoTopDown: list[list[int]] = [ [-1]*200 ] * 20
    memoBottomUp: list[list[int]] = [ [-1]*200 ] * 20
    
    def __init__(self, m, c, garnments) -> None:
        self.garnments: list[list[int]] = garnments
        self.m: int = m
        self.c: int = c


    def weddingTopDown(self):
        """Método que resuelve el problema de Wedding Shopping de forma Top-Down (Recursiva y con Memoisación).
    
        Returns:
            int: Cantidad máxima que se puede gastar comprando una prenda de todas las categorías
        """
        self.memoTopDown = [ [-1]*200 ]*20
        return self.__weddingTopDown(self.m, self.c)


    def __weddingTopDown(self, money: int, c: int):
        if self.c == c:
            # Si ya no alcanzó el dinero en esta ruta, se regresa un negativo
            if money < 0: return -10000
            # Si alcanzó el dinero, obtener el presupuesto total gastado (en la rama)
            else: return self.m - money

        if self.memoTopDown[c][money] > -1:
            return self.memoTopDown[c][money]

        max_budget: int = -1
        # Aplicar el método recursivo a todos los costos para obtener el máximo gasto posible.
        for garnment in self.garnments[c]:
            subproblem = self.__weddingTopDown(money - garnment, c + 1)
            if max_budget < subproblem:
                max_budget = subproblem

        self.memoTopDown[c][money] = max_budget

        return max_budget


    def weddingBottomUp(self):
        """Método que resuelve el problema de Wedding Shopping de forma Bottom Up (Iterativamente y con Tabulación)

        Returns:
            int: Cantidad máxima que se puede gastar comprando una prenda de todas las categorías
        """
        memoBottomUp = [ [-1]*200 ] * 20
        self.__weddingBottomUp(self.m, self.c)


    def __weddingBottomUp(self, money: int, c: int):
        # Aun no existe este problema joven D:
        pass


def pruebaWeddingTopDown():
    """Función que corre 3 casos de prueba para la solución WeddingShopping de forma TopDown 
    """

    # Caso de prueba #1
    program_a = WeddingShopping(m=100, c=4, garnments=[ [8, 6, 4], [5, 10], [1, 3, 3, 7], [50, 14, 23, 8] ])
    a: int = program_a.weddingTopDown()
    assert a == 75 # La solución tiene que ser 75

    # Caso de prueba #2
    program_b = WeddingShopping(m=20, c=3, garnments=[ [4, 6, 8], [5, 10], [1, 3, 5, 5] ])
    b: int = program_b.weddingTopDown()
    assert b == 19 # La solución tiene que ser 19

    # Caso de prueba #3
    program_c = WeddingShopping(m=5, c=3, garnments=[ [6, 4, 8], [10, 6], [7, 3, 1, 7] ])
    c: int = program_c.weddingTopDown()                          
    assert c == -1 # No hay solución


def pruebaWeddingBottomUp():
    """Función que corre 3 casos de prueba para la solución WeddingShopping de forma BottomUp 
    """

    # Caso de prueba #1
    program_a = WeddingShopping(m=100, c=4, garnments=[ [8, 6, 4], [5, 10], [1, 3, 3, 7], [50, 14, 23, 8] ])
    a = program_a.weddingBottomUp()
    assert a == 75 # La solución tiene que ser 75

    # Caso de prueba #2
    program_b = WeddingShopping(m=20, c=3, garnments=[ [4, 6, 8], [5, 10], [1, 3, 5, 5] ])
    b = program_b.weddingBottomUp()
    assert b == 19 # La solución tiene que ser 19

    # Caso de prueba #3
    program_c = WeddingShopping(m=5, c=3, garnments=[ [6, 4, 8], [10, 6], [7, 3, 1, 7] ])
    c = program_c.weddingBottomUp()                          
    assert c == -1 # No hay solución