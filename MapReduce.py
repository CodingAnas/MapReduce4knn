from mrjob.job import MRJob
import math
import heapq

class KNNMapReduce2(MRJob):
    k=3

    def configure_args(self):
        super(KNNMapReduce2, self).configure_args()
        self.add_passthru_arg('--train', type=str, help="Reading the train file")

    def load_train_data(self):
        self.train_data=[]
        with open(self.options.train, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                tup = tuple(map(float, values[:-1]))
                label = values[-1]
                self.train_data.append((tup, label))

    def mapper_init(self):
        self.load_train_data()

    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    
    def mapper(self, _, line):
        points = line.strip().split(',')
        test_point = tuple(map(float, points[:-1])) 
        test_label = points[-1]

        nearest_neighbors = []
        for train_point, label in self.train_data:
            distance = self.euclidean_distance(test_point, train_point)
            heapq.heappush(nearest_neighbors, (-distance, label))
            if len(nearest_neighbors) > self.k :
                heapq.heappop(nearest_neighbors)
        
        for neighbor in nearest_neighbors:
            yield test_label, neighbor


    def reducer(self, test_id, values):
        sorted_values = sorted(values, key=lambda x : abs(x[0]))[:self.k]
        count = {}
        for _, label in sorted_values:
            count[label] = count.get(label, 0) + 1
        
        predicted_label = max(count, key=count.get)
        yield test_id, predicted_label

if __name__ == '__main__':
    KNNMapReduce2.run()

        
