import numpy as np
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1]
with open(file_path, "r") as file:
    line1, *lines = [line.rstrip('\n') for line in file]
    layer_sizes = [ int(i) for i in line1.split("-") ]
    accuracies = []
    zero_rates = []
    group = []
    color_map = {}
    np_colors = ["b", "g", "r", "c", "m", "y", "k", \
                 "#ffa07a",	"#40e0d0", "#7b68ee", "#ff00ff"]
    np_colors.reverse()
    for line in lines:
        if len(line) == 0: pass
        lambda1, accuracy, zero_acts = line.split()
        group.append(lambda1)
        if lambda1 not in color_map:
            color_map[lambda1] = np_colors.pop()
        accuracies.append(float(accuracy))
        layer1, layer2, layer3 = [ float(n) for n in zero_acts.split("-") ]
        # the number of zero activations
        zero_rate = ( (layer_sizes[0] - layer1) + (layer_sizes[1] - layer2)) \
                        / sum(layer_sizes[:-1])
        zero_rates.append(zero_rate)

accuracies = np.array(accuracies)
zero_rates = np.array(zero_rates)
group = np.array(group)
# print(accuracies)
# print(zero_rates)
# print(group)

# regression
x_new = np.linspace(min(zero_rates), max(zero_rates), num=len(zero_rates)*10)
coefs = np.polyfit(zero_rates, accuracies, 2)
ffit = np.polyval(coefs, x_new)

fig, ax = plt.subplots()
ax.plot(x_new, ffit)
plt.title("784-300-100-10 MLP")
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(zero_rates[ix], accuracies[ix], c = color_map[g], label = g)

plt.xlabel("Active activation rate")
plt.ylabel("Accuracy")
print("The number of data: {}".format(len(zero_rates)))
plt.legend()
plt.show()
