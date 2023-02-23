import ggsolver.logic as logic
import ggsolver.logic.products as products


if __name__ == '__main__':
    print(f"Create a PL formula")
    f = logic.PL("(a & b) | b")
    print(f"{f=}")
    print(f"{f.simplify()=}")
    aut = f.translate()
    print(f"{aut=}")

    print()
    print(f"Create a LTL formula")
    f = logic.LTL("F(a & Fb)")
    print(f"{f=}")
    print(f"{f.simplify()=}")
    aut = f.translate()
    print(f"{aut=}")

    print()
    print(f"Create a ScLTL formula")
    f = logic.ScLTL("F(a & Fb)")
    print(f"{f=}")
    print(f"{f.simplify()=}")
    aut0 = f.translate()
    print(f"{aut0=}")

    print()
    print(f"Create a ScLTL formula")
    f = logic.ScLTL("F(a & b)")
    print(f"{f=}")
    print(f"{f.simplify()=}")
    aut1 = f.translate()
    print(f"{aut1=}")

    prod_aut = products.DFAIntersectionProduct([aut0, aut1])
    print(prod_aut.states())


