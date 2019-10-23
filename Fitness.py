
class Fitness:
    def __init__(self, route):
        self.route = route
        self.dist = 0
        self.fit = 0.0 # fitness = 1/distance
    
    def calDistance(self):
        if self.dist ==0:
            totalDist = 0

            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = self.route[i+1]
                if i+1 >= len(self.route):
                    toCity = self.route[0]
                totalDist += fromCity.distance(toCity)
            self.dist = totalDist

        return self.dist
    
    def calFitness(self):
        if self.fit == 0: # why need to check == 0 here (0 means init?)
            self.fit = 1 / float(self.calDistance())

        return self.fit