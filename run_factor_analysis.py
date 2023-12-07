import pandas

from factor_analyzer import FactorAnalyzer 
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

import numpy

dataset = pandas.read_csv("bfi_dataset.csv")

#print(dataset)

#Do a test: We're asking if our matrix is different from an identiy matrix or not

chi2 ,p=calculate_bartlett_sphericity(dataset)
#print(chi2, p)

machine = FactorAnalyzer(n_factors=25, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
#print(ev)

#Gives eigen values for the matrix. 25 columns, 25 eigen values
#Ranks importance. The prinicipal eigen value is the first value. 
#If eigen value is bigger than one, there is an important underlying factor. 
#Six are bigger than 1. 
#Remember we are doing Big Five Inventory
#This test is telling us there are six

machine = FactorAnalyzer(n_factors=6, rotation=None)
machine.fit(dataset)
output = machine.loadings_
#print(output)

#It gives us 25 rows, 6 columns. 
#25 rows are 25 questions.
#Say you answer question 5. The value in the row with the highest number
# will mean that the question is most related with one column, one personality.
# If questions 4 and 5 are calculating the same personality, because their row values are similar
# we can conclude that we don't need to include both questions into our regression. 

machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

#It organizies the questions by their corresponding personality. 
#For example, the first five rows will be similiar and will correspond to one column, by having all the high values in one column. 

#Back to the main reason why we're doing this. 
#We have 4000+ people. We need to assign one personality to all people. 

dataset = dataset.values

results = numpy.dot(dataset, factor_loadings)

pandas.DataFrame(results).round().to_csv("results.csv", index=False)

