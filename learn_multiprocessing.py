import ctypes
import multiprocessing as mp
from queue import Empty


def worker1(pid, queue, count, lock, d):
    with lock:
        print(f"{pid=} started.")

    while count.value > 0:
        # Non-blocking attempt to get next node
        try:
            val = queue.get()
        except Empty:
            continue

        if val < 10:
            if val + 1 not in d:
                queue.put(val + 1)
                d[val+1] = None
                count.value += 1

            if val + 2 not in d:
                queue.put(val + 2)
                d[val+2] = None
                count.value += 1


        # Decrease count when node is processed
        count.value -= 1

        with lock:
            print(f"{pid}, {val}, {count.value}.")

    print(f"{pid=} terminated.")


def main1():
    queue = mp.Queue()
    lock = mp.Lock()
    count = mp.Value(ctypes.c_int16)
    d = mp.Manager().dict()
    processes = dict()

    queue.put(0)
    count.value = 1
    for pid in range(2):
        prc = mp.Process(target=worker1, args=(pid, queue, count, lock, d))
        processes[pid] = prc
        prc.start()

    for prc in processes.values():
        prc.join()

    print(d)


def worker2(pid, queue, count, lock, d):
    with lock:
        print(f"{pid=} started.")

    while True:
        # Blocking attempt to get next node
        val = queue.get()
        if val is None:
            break

        if val in d:
            continue

        d[val] = None

        with lock:
            print(f"{pid}, {val}, {count.value}.")

        if val < 10:
            if val + 1 not in d:
                queue.put(val + 1)
                # d[val+1] = None
                count.value += 1
                with lock:
                    print(f"\t{pid}, ++{val+1}, {count.value}.")

            if val + 2 not in d:
                queue.put(val + 2)
                # d[val+2] = None
                count.value += 1
                with lock:
                    print(f"\t{pid}, ++{val+2}, {count.value}.")
        else:
            queue.put(None)

        # Decrease count when node is processed
        count.value -= 1

    print(f"{pid=} terminated.")


def main2():
    queue = mp.JoinableQueue()
    lock = mp.Lock()
    count = mp.Value(ctypes.c_int16)
    d = mp.Manager().dict()
    processes = dict()

    queue.put(0)
    count.value = 1
    for pid in range(2):
        prc = mp.Process(target=worker2, args=(pid, queue, count, lock, d))
        processes[pid] = prc
        prc.start()

    for prc in processes.values():
        prc.join()

    # Add sentinel to indicate end of data
    queue.put(None)

    queue.join()
    print(d)


def worker3(pid, queue, count, lock, visiting, visited):
    """
    Worker manages 3 objects.
    1. `queue` to get and put nodes to explore.
    2. `visiting`, a dictionary of nodes being processed (ongoing in some process).
    3. `visited`, a dictionary of processed nodes.

    Note: queue may have duplicates, but they every will be visited exactly once.
    """
    with lock:
        print(f"{pid=} started.")

    while True:
        val = queue.get()
        with lock:
            print(f"S {pid}, --{val}, {count.value}.")

        if val is None:
            break

        if val in visiting or val in visited:
            count.value -= 1
            continue

        visiting[val] = None

        # with lock:
        #     print(f"S {pid}, {val}, {count.value}.")

        if val < 10:
            with lock:
                if val + 1 not in visiting and val + 1 not in visited:
                    queue.put(val + 1)
                    # d[val+1] = None
                    count.value += 1
                    # with lock:
                    print(f"A {pid}, ++{val + 1}, {count.value}.")

                if val + 2 not in visiting and val + 2 not in visited:
                    queue.put(val + 2)
                    # d[val+2] = None
                    count.value += 1
                    # with lock:
                    print(f"A {pid}, ++{val + 2}, {count.value}.")
        else:
            queue.put(None)

        # Decrease count when node is processed
        visiting.pop(val)
        visited[val] = None
        count.value -= 1

        with lock:
            print(f"E {pid}, {val}, {count.value}.")

    print(f"{pid=} terminated.")


def main3():
    queue = mp.Queue()
    lock = mp.Lock()
    count = mp.Value(ctypes.c_int16)
    visiting = mp.Manager().dict()
    visited = mp.Manager().dict()
    processes = dict()

    queue.put(0)
    count.value = 1
    for pid in range(2):
        prc = mp.Process(target=worker3, args=(pid, queue, count, lock, visiting, visited))
        processes[pid] = prc
        prc.start()

    for prc in processes.values():
        prc.join()

    # Add sentinel to indicate end of data
    queue.put(None)

    # queue.join()
    print(f"{visiting=}")
    print(f"{visited=}")


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
