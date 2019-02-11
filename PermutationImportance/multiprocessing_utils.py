"""This code is copied nearly wholesale from GrantJ's answer here:
https://stackoverflow.com/questions/5318936/python-multiprocessing-pool-lazy-iteration?noredirect=1&lq=1

It allows for a lazy imap over an iterable and the return of very large objects
"""

from multiprocessing import Process, Queue, cpu_count
try:
    from Queue import Full as QueueFull
    from Queue import Empty as QueueEmpty
except ImportError:  # python3
    from queue import Full as QueueFull
    from queue import Empty as QueueEmpty

__all__ = ["pool_imap_unordered"]


def worker(func, recvq, sendq):
    for args in iter(recvq.get, None):
        result = (args[0], func(*args[1:]))
        sendq.put(result)


def pool_imap_unordered(func, iterable, procs=cpu_count()):
    """
    Lazily imaps in an unordered manner over an iterable in parallel as a 
    generator

    :param func: function to perform on each iterable
    :param iterable: iterable which has items to map over
    :param procs: number of workers in the pool
    :yields: the results of the mapping
    """
    # Create queues for sending/receiving items from iterable.

    sendq = Queue(procs)
    recvq = Queue()

    # Start worker processes.

    for rpt in xrange(procs):
        Process(target=worker, args=(func, sendq, recvq)).start()

    # Iterate iterable and communicate with worker processes.

    send_len = 0
    recv_len = 0
    itr = iter(iterable)

    try:
        value = itr.next()
        while True:
            try:
                sendq.put(value, True, 0.1)
                send_len += 1
                value = itr.next()
            except QueueFull:
                while True:
                    try:
                        result = recvq.get(False)
                        recv_len += 1
                        yield result
                    except QueueEmpty:
                        break
    except StopIteration:
        pass

    # Collect all remaining results.

    while recv_len < send_len:
        result = recvq.get()
        recv_len += 1
        yield result

    # Terminate worker processes.

    for rpt in xrange(procs):
        sendq.put(None)
