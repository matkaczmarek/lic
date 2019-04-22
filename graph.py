class Graph:
    def __init__(self, map):
        self.map = {}
        if type(map) == list:
            for i in range(len(map)):
                self.map[i] = map[i]
        else:
            self.map = map

        self.edge = int(sum(len(value) for key, value in self.map.items()) / 2)

    def __len__(self):
        return len(self.map)

    def __str__(self):
        return str(self.map)

    def add_vert(self, label, neigh):
        self.map[label] = neigh
        for x in neigh:
            self.map[x].append(label)

    def edge_number(self):
        return self.edge

    def verticies(self):
        return list(self.map.keys())

    def neighbours(self, i):
        return self.map[i]

    def subgraph_without(self, X):
        out = {}
        for x in self.map.keys():

            if x not in X:
                out[x] = [i for i in self.map[x] if i not in X]

        return Graph(out)

    def subgraph(self, X):
        out = {}
        for x in self.map.keys():
            if x in X:
                out[x] = [i for i in self.map[x] if i in X]
        return Graph(out)
