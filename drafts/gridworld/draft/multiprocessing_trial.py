from multiprocessing import Process
from time import sleep

"""
Define a class MyApp with run() method. 
The run() method will run instance of MyApp on independent process.
In general, multiple instances of MyApp should be possible to run. 
"""

class MyApp:
    PROCESSES = []
    def __init__(self, value):
        self._value = value
        self._terminate = False

    def _run(self):
        for i in range(10):
            print(i, self._value)
            sleep(1)

    def run(self):
        p = Process(target=self._run)
        p.start()
        # p.join()
        MyApp.PROCESSES.append(p)


if __name__ == '__main__':
    app1 = MyApp("App 1")
    app2 = MyApp("App 2")

    app1.run()
    app2.run()

    while any(p.is_alive() for p in MyApp.PROCESSES):
        pass

    print("OK")
