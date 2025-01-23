from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
model = BayesianNetwork([
    ('Rain', 'Sprinkler'),
    ('Rain', 'WetGrass'),
    ('Sprinkler', 'WetGrass')
])

# Define Conditional Probability Distributions (CPDs)

# Rain probability
cpd_rain = TabularCPD(variable='Rain', variable_card=2, 
                      values=[[0.8], [0.2]])  # 20% chance of rain

# Sprinkler probability (depends on Rain)
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, 
                           values=[[0.6, 0.99],   # No Sprinkler.  The order of probabilities are( rain -> no, yes )
                                   [0.4, 0.01]],  # Sprinkler
                           evidence=['Rain'],
                           evidence_card=[2])  # number of possible states rain=2(yes/no)

# Wet Grass probability (depends on Rain and Sprinkler)
cpd_wet_grass = TabularCPD(variable='WetGrass', variable_card=2,
                           values=[[1.0, 0.1, 0.1, 0.01],  # Not Wet. The order of probabilites are ( sprinker/rain -> off/no, off/yes, on/no, on/yes )
                                   [0.0, 0.9, 0.9, 0.99]], # Wet
                           evidence=['Sprinkler', 'Rain'],
                           evidence_card=[2, 2]) # number of possible states, sprinkler=2(on/off), rain=2(yes/no)

# Add CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)

# Verify the model
print("Model is valid:", model.check_model())

# Perform inference
inference = VariableElimination(model)

# Example queries
print("\nProbability of Wet Grass:")
print(inference.query(variables=['WetGrass']))

print("\nProbability of Wet Grass given Rain:")
print(inference.query(variables=['WetGrass'], evidence={'Rain': 1}))

# Finally plot using networkx
import matplotlib.pyplot as plt 
import networkx as nx 

# Plot the network 
nx.draw(model, with_labels=True, node_size=3000, node_color='lightblue') 
plt.title("Bayesian Network") 
plt.show()