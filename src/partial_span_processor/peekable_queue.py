import queue

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:
            if self._qsize() > 0:
                return self.queue[0]
            return None