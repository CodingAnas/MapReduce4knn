import heapq

numb = []
heapq.heappush(numb, -2)
heapq.heappush(numb, -3)
heapq.heappush(numb, -1)
heapq.heappush(numb, -5)

print(heapq.heappop(numb))