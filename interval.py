from math import inf
from typing import Union

Number = Union[int, float]


class Interval:
    a: Number
    b: Number
    left_open: bool
    right_open: bool
    reals: bool

    @staticmethod
    def empty(reals: bool = False):
        return Interval(0, 0, False, False, reals)

    @staticmethod
    def range(
        l: Number, r: Number, lo: bool = False, ro: bool = True, reals: bool = True
    ):
        return Interval(l, r, lo, ro, reals)

    @staticmethod
    def point(p: Number, lo: bool = False, ro: bool = True, reals: bool = False):
        return Interval(p, p, lo, ro, reals)

    @staticmethod
    def create(
        l: Number, r: Number, lo: bool = False, ro: bool = True, reals: bool = True
    ):
        return Interval(l, r, lo, ro, reals)

    @staticmethod
    def open(l: Number, r: Number, reals: bool = True):
        return Interval(l, r, True, True, reals)

    @staticmethod
    def closed(l: Number, r: Number, reals: bool = True):
        return Interval(l, r, False, False, reals)

    @staticmethod
    def left_open(l: Number, r: Number, reals: bool = True):
        return Interval(l, r, True, False, reals)

    @staticmethod
    def right_open(l: Number, r: Number, reals: bool = True):
        return Interval(l, r, False, True, reals)

    @staticmethod
    def reals(reals: bool = True):
        return Interval(-inf, inf, True, True, reals)

    def __init__(
        self,
        l: Number,
        r: Number,
        lo: bool = False,
        ro: bool = True,
        reals: bool = True,
    ):
        self.a = l
        self.b = r
        self.left_open = lo
        self.right_open = ro
        self.reals = reals

    def is_empty(self):
        return self.a > self.b or (
            self.a == self.b and (self.left_open or self.right_open)
        )

    def left(self):
        return self.a

    def right(self):
        return self.b

    def __str__(self):
        s = "(" if self.left_open else "["

        if self.a == -inf:
            s += "-∞"
        else:
            s += str(self.a)
        s += ", "
        if self.b == inf:
            s += "∞"
        else:
            s += str(self.b)

        s += ")" if self.right_open else "]"

        return s
