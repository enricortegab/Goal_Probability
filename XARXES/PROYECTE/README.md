# Anàlisi i Filtrat de Frames per a l’Estimació de Goal Probability mitjançant Xarxes Neuronals

Aquest projecte té com a objectiu analitzar i classificar frames just abans del moment del xut en partits de futbol per a estimar la probabilitat de gol. La principal tasca és filtrar frames de qualitat per a una posterior anàlisi i predicció fiable.

---

## **Descripció**

El treball es basa en l’ús de models de detecció d’objectes YOLO per identificar jugadors, pilota i porteries en frames capturats abans del xut. Amb aquesta informació, es genera un embedding que permet classificar els frames en bons o dolents mitjançant un algorisme no supervisat (k-means).

Per a l’estimació de la probabilitat de gol, s’han utilitzat tres arquitectures de xarxes neuronals diferents: CNN, CNN+LSTM i CNN+Attention.

Aquestes arquitectures s’analitzen i comparen per tal de determinar la millor aproximació per predir el goal probability.




---

## **Requisits previs**

### **Dependències necessàries**

* L’entorn utilitzat ha estat Python.  
* És necessari tenir instal·lades les llibreries que apareixen en el requirements.txt.



## **Instruccions d'ús**

### **Execució del projecte**


1. **Preparació de les deteccions i embeddings**  
   Primer, s’han d’executar els models YOLO per detectar jugadors, pilota i porteria en cada frame.  
   Aquesta informació s’emmagatzema en forma de *bounding boxes* i scores de confiança.  
   Amb aquestes dades, es genera un embedding per a cada frame, tenint en compte les condicions definides (detecció correcta de pilota i porteria, proximitat jugador-pilota, orientació del jugador).  

2. **Classificació amb k-means**  
   Un cop calculats els embeddings, s’aplica un model k-means amb 2 classes per separar els frames bons dels dolents de manera no supervisada.  
   Per reforçar la separació, quan falti la detecció o el jugador estigui de l’esquena, s’assigna un valor elevat en l’embedding per facilitar la classificació.  

3. **Anàlisi dels resultats**  
   El sistema permet validar la qualitat de la classificació i visualitzar el rendiment dels models YOLO i k-means.  
   Els frames classificats com a bons es poden utilitzar per a l’anàlisi posterior del goal probability.

---

## **Contacte**

Per a dubtes, suggeriments o sol·licituds de dades i models:  
- **Nom**: [David Araujo García],[Eloi Mercader Morillas],[Enric Ortega Barreda] i [Joan Tubert Mascort]  
- **Email**: [1671077@uab.cat], [1666675@uab.cat], [1672973@uab.cat] i [1673326@uab.cat]  
