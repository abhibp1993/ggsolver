# from ggsolver.logic.pl import *
from ggsolver.logic import *


if __name__ == '__main__':
    # f0 = PL("a & b")
    f0 = PL("a")
    print(f0, f0.atoms(), f0.evaluate({"a"}))

    f1 = PL("a | b", atoms={"a", "b", "c"})
    print(f1, f1.atoms(), f1.evaluate({"a", "c"}))

    f2 = PL("a & b", atoms={"c"})
    print(f2, f2.atoms(), f2.evaluate({"a", "c"}), f2.evaluate({"c"}))

    try:
        f3 = PL("Fa & b")
    except ParsingError:
        print("ParsingError test OK.")

