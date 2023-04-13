import ctypes
import multiprocessing as mp
from queue import Empty


def worker4(pid, queue, count, lock, queue_dict, visited):
    with lock:
        print(f"{pid=} started.")

    while True:
        # Termination condition
        if count.value == 0:
            break

        # Get next element
        try:
            val = queue.get(timeout=0.01)
        except Empty:
            continue

        with lock:
            print(f"P {pid} --{val}, {list(queue_dict.items())}, {list(visited.items())}")

        # If sentinel, terminate
        if val is None:
            break

        # Visit element (update dQ, V)
        with lock:
            queue_dict.pop(val)
            visited[val] = None

        with lock:
            print(f"V {pid} --{val}")

        # Generate new elements
        new_values = [val + 1, val + 2]

        # For each element, if it is in dQ or V, do nothing. Else add to Q, dQ. Increment count.
        with lock:
            for n_val in new_values:
                if n_val > 10:
                    continue
                if not (n_val in queue_dict or n_val in visited):
                    queue.put(n_val)
                    queue_dict[n_val] = None
                    count.value += 1
                    # with lock:
                    print(f"A {pid} ++{n_val}")

        # Decrement count by 1.
        count.value -= 1
        print(f"F {pid} {list(queue_dict.items())}, {list(visited.items())} {count.value}")

    print(f"{pid=} terminated.")


def main4():
    queue = mp.Queue()
    lock = mp.Lock()
    count = mp.Value(ctypes.c_int16)
    queue_dict = mp.Manager().dict()
    visited = mp.Manager().dict()
    processes = dict()

    # Initialize shared variables
    queue.put(0)
    queue_dict[0] = None
    count.value = 1

    for pid in range(2):
        prc = mp.Process(target=worker4, args=(pid, queue, count, lock, queue_dict, visited))
        processes[pid] = prc
        prc.start()

    for prc in processes.values():
        prc.join()

    # Add sentinel to indicate end of data
    queue.put(None)

    # queue.join()
    print(f"{list(queue_dict.items())=}")
    print(f"{list(visited.items())=}")


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    main4()
