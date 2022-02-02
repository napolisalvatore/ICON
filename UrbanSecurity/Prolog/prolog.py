# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:45:37 2021

@author: napol
"""
import pytholog as pl
new_kb = pl.KnowledgeBase("Informatica")

#base di conoscenza: insieme di asserzioni
new_kb([
                                                   
        "osserva (telecamera1, marciapiede_viaZanardelli)",        #osserva(telecamera,luogo)
        "osserva (telecamera2, strada_viaOrabona)",
        "osserva (telecamera3, telecaera2)",
        "osserva (telecamera4, statua_piazzaMercantile)",
        "osserva (telecamera5, piazza_Mercantile)",
        "osserva (telecamera6, marciapiede_viaPasubio)",       
        "osserva (telecamera7, strada_viaArgiro)",
        "osserva (telecamera8, telecaera5)",
        "osserva (telecamera7, strada_Imera)",
        "osserva (telecamera8, largo_albicocca)",
        "osserva (telecamera9, largo_ciaia)",
        
        #data in cui si è verificata violenza
        "data (telecamera1, 12/02/2018)",       #data(telecamera, data scena di violenza)
        "data (telecamera1, 13/05/2018)",
        "data (telecamera1, 14/01/2019)",
        "data (telecamera1, 06/04/2019)",
        "data (telecamera1, 31/05/2019)",
        "data (telecamera1, 27/11/2019)",
        "data (telecamera1, 24/12/2020)",
        "data (telecamera2, 03/09/2018)",
        "data (telecamera2, 29/04/2021)",
        "data (telecamera2, 01/07/2021)",
        "data (telecamera2, 14/09/2018)",
        "data (telecamera2, 30/12/2021)",
        "data (telecamera2, 11/01/2019)",
        "data (telecamera3, 01/06/2019)",
        "data (telecamera3, 30/11/2020)",
        "data (telecamera3, 06/01/2019)",
        "data (telecamera3, 23/07/2021)",
        "data (telecamera3, 24/10/2020)",
        "data (telecamera3, 11/01/2020)",
        "data (telecamera4, 14/03/2021)",
        "data (telecamera4, 10/11/2021)",
        "data (telecamera4, 11/05/2020)",
        "data (telecamera4, 30/03/2019)",
        "data (telecamera4, 27/08/2021)",
        "data (telecamera4, 22/11/2020)",
        "data (telecamera4, 10/11/2020)",
        "data (telecamera4, 15/09/2019)",
        "data (telecamera5, 30/10/2021)",
        "data (telecamera5, 23/07/2021)",
        "data (telecamera5, 24/03/2021)",
        "data (telecamera5, 28/06/2020)",
        "data (telecamera5, 14/09/2019)",
        "data (telecamera5, 23/08/2021)",
        "data (telecamera5, 22/06/2020)",
        "data (telecamera5, 27/12/2018)",
        "data (telecamera6, 23/04/2020)",
        "data (telecamera6, 06/02/2020)",
        "data (telecamera6, 07/08/2020)",
        "data (telecamera6, 19/06/2020)",
        "data (telecamera6, 11/01/2019)",
        "data (telecamera6, 20/01/2021)",
        "data (telecamera7, 08/09/2020)",
        "data (telecamera7, 07/09/2021)",
        "data (telecamera7, 23/07/2019)",
        "data (telecamera7, 13/08/2020)",
        "data (telecamera7, 04/10/2021)",
        "data (telecamera7, 12/11/2021)",
        "data (telecamera8, 04/12/2021)",   
        "data (telecamera8, 29/09/2020)", 
        "data (telecamera9, 21/11/2020)", 
        "data (telecamera9, 05.01.21)", 
        "data (telecamera9, a29/06/2020)",
        

        
        #numero di persone cinvolte
        "num (12/02/2018,2)",              #num(data, personeCoinvolte)
        "num (14/01/2019,3)",
        "num (06/04/2019,6)",
        "num (31/05/2019,9)",
        "num (27/11/2019,3)",
        "num (24/12/2020,5)",
        "num (03/09/2018,9)",
        "num (29/04/2021,4)",
        "num (01/07/2021,2)",
        "num (14/09/2018,7)",
        "num (30/12/2021,6)",
        "num (11/01/2021,5)",
        "num (01/06/2019,3)",
        "num (30/11/2020,6)",
        "num (06/01/2019,2)",
        "num (23/07/2021,9)",
        "num (24/10/2020,3)",
        "num (11/01/2020,7)",
        "num (14/03/2021,8)",
        "num (10/11/2021,4)",
        "num (11/05/2020,3)",
        "num (30/03/2019,11)",
        "num (27/08/2021,3)",
        "num (22/11/2020,7)",
        "num (10/11/2020,3)",
        "num (15/09/2019,4)",
        "num (30/10/2021,8)",
        "num (23/07/2021,9)",
        "num (24/03/2021,4)",
        "num (28/06/2020,5)",
        "num (14/09/2019,7)",
        "num (23/08/2021,8)",
        "num (22/06/2020,4)",
        "num (27/12/2018,5)",
        "num (23/04/2020,6)",
        "num (06/02/2020,2)",
        "num (07/08/2020,3)",
        "num (19/06/2020,4)",
        "num (11/01/2019,2)",
        "num (20/01/2021,4)",
        "num (08/09/2020,6)",
        "num (07/09/2021,3)",
        "num (23/07/2019,5)",
        "num (13/08/2020,3)",
        "num (04/10/2021,6)",
        "num (12/11/2021,9)",
        "num (04/12/2021,11)",   
        "num (29/09/2020,5)", 
        "num (21/11/2020,3)", 
        "num (05/01/2021,5)", 
        "num (29/06/2020,4)",
        
        
        #Violenza(IdTelecamera,Luogo,Data,Persone coinvolte):-osserva(IdTelecamera,Luogo),data(IdTelecamera,Data),NumPersoneCoinvolte(Data,nPersone)
        "Violenza(T,L,D,P):-osserva(T,L),data(T,D),num(D,P)"])  

def ask_KB(new_kb, x):
    results=set()
    print("Query: " + str(x))
    y=new_kb.query(pl.Expr(x))
    

    print("Risultati\n", y)
    # Elimina i duplicati dai risultati
    s = ""
    for e in y:
        s = str(e)
        results.add(s)
    return results

#Esempio di utilizzo:

print(ask_KB(new_kb,"osserva(telecamera9,L)"))

