# Projet DL : Normalizing Flow

Ce repository contient le code servant au fonctionnement de l'application Streamlit associée à notre projet. Vous pourrez y trouver une implémentation de notre modèle NICE sur le dataset MNIST. Vous pourrez modifier les paramètres du modèle et observer les résultats : la log likelihood sur les datasets d'entraînement, de validation et de test, ainsi que les images 'de type MNIST' générées par la fonction inverse du modèle. 

Le script Model_NICE.py contient notre code du modèle ; le script app.py y fait appel et fait tourner l'application. 

Les packages nécéssaires sont : 
- os
- streamlit
- pandas
- numpy
- matplotlib.pyplot
- torch

- Justine Lesecq et Maëlle Sifferlin 
