class Fraction:
    def __init__(self, num, denom):
        self.num = num
        self.denom = denom

    def __iter__(self):
        for i in [self.num, self.denom]:
            yield i

    def __str__(self):
        return f"{self.num}/{self.denom}"

    # def __repr__(self):
    #     return str((self.num, self.denom))

    def __eq__(self, other):
        if isinstance(other, Fraction):
            return self.num == other.num and self.denom == other.denom
        return False

    def __hash__(self):
        return hash((self.num, self.denom))

    # TODO: hacky, won't actually work for numerical comparisons, but should work for sorting
    def __lt__(self, other):
        return str(self) < str(other)


import operator


class FractionProblem:
    def __init__(self, frac1: Fraction, frac2: Fraction, str_op: str):
        self.frac1 = frac1
        self.frac2 = frac2
        valid_ops = ["*", "+"]
        if str_op not in valid_ops:
            raise ValueError(f"{str_op} must be in {valid_ops}")

        self.str_op = str_op

        if str_op == "+":
            self.op = operator.add
        elif str_op == "*":
            self.op = operator.mul
        else:
            raise ValueError()

    def __eq__(self, other):
        if isinstance(other, FractionProblem):
            return (
                self.frac1 == other.frac1
                and self.frac2 == other.frac2
                and self.str_op == other.str_op
            )
        return False

    def __hash__(self):
        return hash((self.frac1, self.frac2, self.str_op))

    def __str__(self):
        return f"{self.frac1}{self.str_op}{self.frac2}"

    # def __repr__(self):
    #     return str(self)


def parse_frac(frac_str: str):
    op = "*" if "*" in frac_str else "+"
    frac1, frac2 = frac_str.split(op)
    num1, denom1 = frac1.split("/")
    num2, denom2 = frac2.split("/")
    frac1 = Fraction(int(num1), int(denom1))
    frac2 = Fraction(int(num2), int(denom2))
    return FractionProblem(frac1, frac2, op)
