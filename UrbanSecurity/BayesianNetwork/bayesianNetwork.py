# -*- coding: utf-8 -*-
"""
@author: salvo
"""



from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination



'''
Funzione per fare inferenza sulla violenza in base ai dati passati in input
sotto forma di dizionario (Feature, Valore)
'''
def gradeBayesianInference(evidences):
grades_infer = VariableElimination(grades)
viol = grades_infer.query(variables = ['Violenza'], evidence = evidences)
return viol



'''
Creazione della Rete Bayesiana
'''
grades =BayesianNetwork([('Posizione', 'Luogo'),
('Luogo', 'Violenza'),
('Giorno', 'Violenza'),
('Ora', 'Violenza'),
('Mole di Persone', 'Violenza')
])



'''
Creazione delle varie tabelle di distribuzione di probabilità
'''
location_cpd = TabularCPD('Posizione', 2, [[0.35],[0.65] ])



place_cpd= TabularCPD('Luogo', 3, [[0.20, 0.30],
[0.30, 0.30],
[0.50, 0.40]],
evidence=['Posizione'], evidence_card=[2])



day_cpd = TabularCPD('Giorno', 2, [[0.35], [0.65]])



time_cpd = TabularCPD('Ora', 2, [[0.35], [0.65]])



massesOfPeople_cpd = TabularCPD('Mole di Persone', 3, [[0.20], [0.35], [0.45]])



grade_cpd = TabularCPD('Violenza', 3, [[0.60,0.70,0.85,0.80,0.90,0.95,0.15,0.35,0.60,0.25,0.43,0.65,0.50,0.45,0.55,0.35,0.41,0.65,0.17,0.25,0.35,0.10,0.19,0.22,0.10,0.15,0.18,0.20,0.22,0.25,0.07,0.06,0.05,0.06,0.10,0.15],
[0.35,0.27,0.14,0.17,0.08,0.04,0.70,0.55,0.35,0.65,0.50,0.32,0.30,0.40,0.35,0.55,0.53,0.32,0.63,0.60,0.55,0.40,0.43,0.48,0.50,0.58,0.60,0.46,0.55,0.58,0.10,0.30,0.50,0.27,0.40,0.45],
[0.05,0.03,0.01,0.03,0.02,0.01,0.15,0.10,0.05,0.10,0.07,0.03,0.20,0.15,0.10,0.10,0.06,0.03,0.20,0.15,0.10,0.50,0.38,0.30,0.40,0.27,0.22,0.34,0.23,0.17,0.83,0.64,0.45,0.67,0.50,0.40],
], evidence=['Luogo', 'Giorno', 'Ora', 'Mole di Persone'], evidence_card=[3, 2, 2, 3])




grades.add_cpds(location_cpd, place_cpd, day_cpd, time_cpd, massesOfPeople_cpd, grade_cpd)



# Controlla che le probabilità siano state impostate correttamente
if not grades.check_model():
print("Errore nella generazione della Rete Bayesiana!")
exit


# Esempio:
print(gradeBayesianInference({'Luogo':1}))