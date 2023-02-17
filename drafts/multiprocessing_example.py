import multiprocessing
# import cProfile
# import pstats
# import io


def worker_multi(start, end, result):
    sum = 0
    for i in range(start, end):
        sum += i
    result.update({sum: None})


def worker_single(start, end):
    sum = 0
    for i in range(start, end):
        sum += i
        if i % 1000000 == 0:
            print(i)
    # result.update({sum: None})


def main1(N):
    NUM_PROCESSES = multiprocessing.cpu_count()
    print(NUM_PROCESSES)
    CHUNKSIZE = N // NUM_PROCESSES
    manager = multiprocessing.Manager()
    result = manager.dict()
    jobs = []
    for i in range(NUM_PROCESSES):
        start = i * CHUNKSIZE
        end = start + CHUNKSIZE
        p = multiprocessing.Process(target=worker_multi, args=(start, end, result))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    print(set(result.keys()))


if __name__ == "__main__":
    N = int(5 * 1e8)
    print("parallel")
    main1(N)

    print("single-core")
    worker_single(1, N)
