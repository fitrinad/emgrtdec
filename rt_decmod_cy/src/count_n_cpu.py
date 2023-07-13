from multiprocessing import cpu_count

workers = cpu_count()
print(workers)