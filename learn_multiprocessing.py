import ctypes
import multiprocessing as mp


def gen_edges_mc(data, queue, pid, prc_state):
    print(pid, " started")
    while True:
        if queue.empty():
            print(pid, "empty.")
            prc_state[pid] = 0
            if all(v == 0 or v == 2 for v in prc_state):
                break
        else:
            prc_state[pid] = 1
            curr_state = queue.get()
            print(pid, curr_state)
            # apply delta to generate edges
            if curr_state < 100:
                queue.put(curr_state + 1)

    prc_state[pid] = 2


def producer(queue, state):
    for i in range(10):
        if state.value == 0:
            queue.put(i)
        else:
            break

    state.value = 1


def consumer(queue, state):
    while True:
        if state.value == 0:
            continue

        if queue.empty():
            break

        item = queue.get()
        print(f"Got item: {item}")

    state.value = 2


def main():
    queue = mp.Queue()
    queue.put(0)
    state = mp.Array(ctypes.c_int, 2)
    for i in range(len(state)):
        state[i] = -1

    pid1 = mp.Process(target=gen_edges_mc, args=(None, queue, 0, state))
    pid2 = mp.Process(target=gen_edges_mc, args=(None, queue, 1, state))

    pid1.start()
    pid2.start()
    pid1.join()
    pid2.join()


if __name__ == '__main__':
    main()

    # queue = mp.Queue()
    # state = mp.Value('i', 0)
    #
    # producer_process = mp.Process(target=producer, args=(queue, state))
    # consumer_process = mp.Process(target=consumer, args=(queue, state))
    #
    # producer_process.start()
    # consumer_process.start()
    #
    # producer_process.join()
    # consumer_process.join()
