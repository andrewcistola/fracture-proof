pop = [0.0]
inc = [0.0]
p = 0.0
i = 0.0
for j in range(num):
    p += freq_data[j]
    i += freq_data[j]*inc_data[j]
    pop.append((p/total_pop) * 100)
    inc.append((i/total_inc) * 100)

# print pop
# print inc
# print i

# Preparing data set to plot Lorenz curve
dataset = []

n = len(inc)

for i in range(n):
    dataset.append((pop[i], inc[i]))

lineEquality = [(0, 0), (100, 100)]

# Plotting Lorenz Curve
simpleplot.plot_lines('Lorenz curve', 500, 500, 'population percent', 'income percent', [dataset, lineEquality], True)

# Finding Gini Coefficient
def gini(pop, inc):
    n = len(pop)
    a = 0
    p = pop[0]
    i = inc[0]
    for t in range(1,n):
        a  += (100-p)*(inc[t]-i) - ((pop[t]-p)*(inc[t]-i)/2)
        p = pop[t]
        i = inc[t]
    eq = (100.0*100.0)/2
    return  (eq - a)/eq
    
print "Gini Coefficient is " + str(gini(pop, inc))