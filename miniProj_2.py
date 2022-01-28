import math
import numpy as np

m = 2500
C = 5.5

class Movie:
    def __init__(self, id,avgrate,votes) -> None:
        self.id = id
        self.avgrate = avgrate
        self.votes = votes
        self.imrate = None

    def __str__(self) -> str:
        return "Id :{} || Rating : {} || Votes : {} || imRate : {}".format(self.id,self.avgrate, self.votes,self.imrate)

    ## Function to calculate Rating
    def calcRank(self):
        rating = ((self.votes * self.avgrate)/(self.votes + m)) + ((m*C)/(m+self.votes))
        self.imrate = rating
        return self.imrate



## __main__

bolly = np.genfromtxt("bollywood.csv", delimiter=',', dtype=str)

ranklist = []

for i in range(1, len(bolly)):
    mov = Movie(bolly[i][0],float(bolly[i][1]),float(bolly[i][2]))
    mov.calcRank()
    #print(mov)
    ranklist.append(mov)

## Sorting by rating
ranklist.sort(reverse=True, key=lambda m : m.imrate)

print("\n THE RANKLIST \n")

## Printing the top 10 movies.
for i in range(10):
    print("Rank : {} || imdb_id : {} || Rating : {}".format(i+1, ranklist[i].id,ranklist[i].imrate))