import queue

class CustomPQ(queue.PriorityQueue):
    def peek(self):
        # Access the internal queue list directly
        with self.mutex:
            if self.queue:
                return self.queue[0]  # Peek at the front element without removing
            return None