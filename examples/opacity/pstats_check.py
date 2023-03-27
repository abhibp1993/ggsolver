import pstats

if __name__ == '__main__':

    # create a Stats object from the profiling results
    stats = pstats.Stats('my_profile_results.txt')

    # print the top 10 functions by total time
    stats.strip_dirs().sort_stats('tottime').print_stats(20)
