# Solved with Gurobi 8.0.1
# Requires Python 2.7

import math
import csv
from itertools import combinations, permutations
from gurobipy import *


def euclideanDistance(point1, point2):
    return math.sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def constructModel():
    m = Model()
    # Create binary variables
    vars = {}
    for i in range(81):
        for j in range(81):
            vars[i,j] = m.addVar(obj=c[i,j], vtype=GRB.BINARY,
                            name="x[{0}][{1}]".format(i,j))
        m.update()
    # Incoming and outgoing constraints
    for i in range(81):
        m.addConstr(quicksum(vars[i,j] for j in range(81)) == 1)
        vars[i,i].ub = 0
    for j in range(81):
        m.addConstr(quicksum(vars[i,j] for i in range(81)) == 1)
    m.update()
    return m, vars

def addSubtourConstraints(model, xs, subtours):
    for tour in subtours:
        s = len(tour.keys()) - 1
        model.addConstr(quicksum(xs[ind] for ind in list(tour.keys())) <= s)
    model.update()

def solveModel(model, xs):
    model.optimize()
    solution = model.getAttr('x', xs)
    # Get visited pairs
    pairs = []
    for i in range(81):
        for j in range(81):
            if solution[i,j] > 0.5:
                pairs.append([i,j])
    return m.objVal, pairs

def writePairsSolution(pairs, filename = 'solver_solution.csv'):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for p in pairs:
            writer.writerow(p)


# Data read operation
tsp_patterns = []
with open('IE440Final19ETSPData.txt', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        tsp_patterns.append([float(row["x"]),float(row["y"])])

# Cost calculation
c = {}

for i in range(81):
    for j in range(81):
        c[i,j] = euclideanDistance(tsp_patterns[i],tsp_patterns[j])

# Constructing the model and solving it
m, xs = constructModel()
obj, pairs = solveModel(m, xs)
pairs.sort()
# Find subroutes after each solution and
# add them to constraints until no subroute remains
while True:
    indices = {}
    for i in range(len(pairs)):
        indices[i] = 1
    subtours = []
    for k in range(len(indices)):
        if indices[k] == 0:
            continue
        tour = {}
        t = 0
        i = pairs[k][0]
        while t == len(tour.keys()):
            j = pairs[i][1]
            tour[i,j] = 1
            t+=1
            indices[i] = 0
            i = j
        if len(tour.keys()) < 81:
            subtours.append(tour)
    addSubtourConstraints(m, xs, subtours)
    obj, pairs = solveModel(m, xs)
    pairs.sort()
    if len(subtours) == 0:
        break

# Extract results as a csv file
writePairsSolution(pairs)