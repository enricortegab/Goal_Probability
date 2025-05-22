# Conclusions Generals

Durant el desenvolupament d’aquest projecte, hem pogut extreure diverses conclusions importants que reflecteixen tant els reptes trobats com els resultats obtinguts.

## Dataset i qualitat de dades

El dataset utilitzat presenta un desbalance notable, amb una major proporció de casos no gol (0) respecte als gols (1). Aquest desbalanceig ha tingut un impacte significatiu en el rendiment dels models, ja que, tot i obtenir una precisió aparentment alta, aquesta es deu principalment a la preponderància dels exemples negatius. Això ha dificultat l’aprenentatge efectiu del model en la classificació correcta dels gols.

A més, la qualitat de les imatges i la seva heterogeneïtat ha estat un altre factor limitant. La presència de frames de baixa qualitat o inadequats ha afegit complexitat al procés d’extracció de característiques rellevants, malgrat els esforços realitzats per filtrar-los mitjançant la detecció amb models YOLO i la classificació amb k-means.

## Rendiment del model i aprenentatge

Els resultats mostren un mal aprenentatge general, amb una millora escassa o nul·la malgrat augmentar el nombre d’èpoques durant l’entrenament. Aquest fenomen indica que el model pot estar estancat, possiblement a causa de la qualitat i la distribució de les dades, així com a la dificultat d’aprendre patrons clars a partir de la informació disponible.

Un altre punt destacat ha estat la comparació entre funcions de pèrdua. La funció Binary Cross Entropy (BCE) ha demostrat ser més efectiva que la funció de pèrdua pròpia dissenyada, que intentava centrar l’aprenentatge exclusivament en el centre de la imatge per focalitzar-se en les zones més rellevants. Aquesta constatació indica que, encara que la idea era interessant, l’enfocament tradicional resulta més robust en aquest cas.

## Millores i variables rellevants

Una millora significativa s’ha observat quan s’ha inclòs el coeficient TM, que mesura la pressió de l’equip rival sobre el xutador en el moment immediatament anterior al xut. Aquest factor contextual ha ajudat a enriquir el model amb informació clau que no es reflectia només amb les imatges, millorant així la capacitat predictiva.

## Limitacions tècniques i temporals

El projecte s’ha vist condicionat per limitacions tècniques importants, especialment pel rendiment i disponibilitat de la GPU i servidor proporcionats per l’equip docent. Aquestes dificultats han restringit la capacitat d’entrenar models més complexos o amb major volum de dades, limitant així l’abast i la profunditat de l’anàlisi.

## Conclusió final

En conclusió, el projecte ha estat un repte ambiciós tenint en compte el temps disponible i les limitacions materials i de dades. El dataset ha estat limitat tant en quantitat com en qualitat, amb un fort desbalanceig i imatges no òptimes, fet que ha afectat el rendiment dels models i la seva capacitat d’aprenentatge.

Malgrat tot, s’ha aconseguit establir una base sòlida i identificar àrees clau per a futures millores, com ara la incorporació de variables contextuals (com el coeficient TM) i la necessitat d’obtenir datasets més equilibrats i de millor qualitat.

