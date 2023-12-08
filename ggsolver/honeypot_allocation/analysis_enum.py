import pickle


def main():
    # Load the pickle file
    with open("out/gw2_t1_f1_enum/dswin_enumerative_results.pkl", "rb") as file:
        data = pickle.load(file)

    for key, val in data.items():
        print(key, val)

    # Identify maximum value elements in data
    max_val = max(data.values())
    max_keys = [k for k, v in data.items() if v == max_val]
    print(round(max_val, 2), max_keys)


if __name__ == '__main__':
    main()
