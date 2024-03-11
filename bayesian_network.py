from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import TabularCPD


def show_bn(model):
	pgm_graph = nx.DiGraph()
	pgm_graph.add_edges_from(model.edges())
	pos = nx.spring_layout(pgm_graph, seed=46)
	plt.figure(figsize=(10, 6))
	nx.draw(pgm_graph, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=12, font_color='black')
	plt.title("Bayesian Network")
	plt.show()


model_attr = [('Construction', 'Traffic'),
			  ('Rain', 'Traffic'),
			  ('Rain', 'Umbrella')]

model = BayesianNetwork(model_attr)

cpd_construction = TabularCPD(variable='Construction', variable_card=2,
							  values=[[0.7], [0.3]])
cpd_rain = TabularCPD(variable='Rain', variable_card=2,
					  values=[[0.4], [0.6]])
cpd_umbrella = TabularCPD(variable='Umbrella', variable_card=2,
						  values=[[0.9, 0.27], [0.1, 0.73]],
						  evidence=['Rain'], evidence_card=[2])
cpd_traffic = TabularCPD(variable='Traffic', variable_card=2,
						 values=[[0.999, 0.71, 0.6, 0.41],
								 [0.001, 0.29, 0.4, 0.59]],
						 evidence=['Construction', 'Rain'],
						 evidence_card=[2, 2])

model.add_cpds(cpd_construction, cpd_rain, cpd_umbrella, cpd_traffic)

print('The cpds are valid for the model:', model.check_model())
print('Model enodes:', model.nodes())
print('Model edges:', model.edges())
print('Checking independence of Construction', model.local_independencies('Construction'))
print('Checking independence of Rain', model.local_independencies('Rain'))
print('Checking independence of Umbrella', model.local_independencies('Umbrella'))
print('Checking independence of Traffic', model.local_independencies('Traffic'))
print('All Independencies', model.get_independencies())

show_bn(model)

model_infer = VariableElimination(model)
q = model_infer.query(variables=["Traffic"], evidence={"Construction": 0})
print(q)
q = model_infer.query(variables=["Rain"], evidence={"Traffic": 1, "Construction": 0})
print(q)
q = model_infer.query(variables=["Rain", "Traffic"])
print(q)
