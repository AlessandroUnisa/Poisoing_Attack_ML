import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import datasets as dt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pandas as pd
import numpy as np
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import   matplotlib


from skimage.io import imread, imshow

from attack import X_train, y_train, X_test, y_test, targeted_gradient, non_targeted_gradient


class Attack:

    def __init__(self, model):
        self.fooling_targets = None
        self.model = model


    def prepare(self, X_train, y_train, X_test, y_test):
        self.images = X_test
        self.true_targets = y_test
        self.num_samples = X_test.shape[0]
        self.train(X_train, y_train)
        print("Model training finished.")
        self.test(X_test, y_test)
        print("Model testing finished. Initial accuracy score: " + str(self.initial_score))

    def set_fooling_targets(self, fooling_targets):
        self.fooling_targets = fooling_targets

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.weights = self.model.coef_
        self.num_classes = self.weights.shape[0]

    def test(self, X_test, y_test):
        self.preds = self.model.predict(X_test)
        self.preds_proba = self.model.predict_proba(X_test)
        self.initial_score = accuracy_score(y_test, self.preds)

    def create_one_hot_targets(self, targets):
        self.one_hot_targets = np.zeros(self.preds_proba.shape)
        for n in range(targets.shape[0]):
            self.one_hot_targets[n, targets[n]] = 1

    def attack(self, attackmethod, epsilon):
        perturbed_images, highest_epsilon = self.perturb_images(epsilon, attackmethod)
        perturbed_preds = self.model.predict(perturbed_images)
        score = accuracy_score(self.true_targets, perturbed_preds)
        return perturbed_images, perturbed_preds, score, highest_epsilon

    def perturb_images(self, epsilon, gradient_method):
        perturbed = np.zeros(self.images.shape)
        max_perturbations = []
        for n in range(self.images.shape[0]):
            perturbation = self.get_perturbation(epsilon, gradient_method, self.one_hot_targets[n], self.preds_proba[n])
            perturbed[n] = self.images[n] + perturbation
            max_perturbations.append(np.max(perturbation))
        highest_epsilon = np.max(np.array(max_perturbations))
        return perturbed, highest_epsilon

    def get_perturbation(self, epsilon, gradient_method, target, pred_proba):
        gradient = gradient_method(target, pred_proba, self.weights)
        inf_norm = np.max(gradient)
        perturbation = epsilon / inf_norm * gradient
        return perturbation

    def attack_to_max_epsilon(self, attackmethod, max_epsilon):
        self.max_epsilon = max_epsilon
        self.scores = []
        self.epsilons = []
        self.perturbed_images_per_epsilon = []
        self.perturbed_outputs_per_epsilon = []
        for epsilon in range(0, self.max_epsilon):
            perturbed_images, perturbed_preds, score, highest_epsilon = self.attack(attackmethod, epsilon)
            self.epsilons.append(highest_epsilon)
            self.scores.append(score)
            self.perturbed_images_per_epsilon.append(perturbed_images)
            self.perturbed_outputs_per_epsilon.append(perturbed_preds)

    def run_attack(self, attack_type, epsilon):
        if attack_type == "non_targeted":
            # Esegui l'attacco non mirato
            return self.attack(non_targeted_gradient, epsilon)
        elif attack_type == "targeted":
            # Esegui l'attacco mirato
            return self.attack(targeted_gradient, epsilon)
        else:
            raise ValueError("Tipo di attacco non supportato")

class DefenseAttack(Attack):
    def _init_(self, model):
        super()._init_(model)

    def defense_method(self, X):
        # Applica Principal Component Analysis (PCA) per ridurre la dimensionalità
        num_components = 50  # Scegli il numero di componenti principali
        pca = PCA(n_components=num_components)
        reduced_X = pca.fit_transform(X)

        # Inverti la trasformazione per ottenere le immagini difese
        defended_X = pca.inverse_transform(reduced_X)

        return defended_X

    def prepare_defense(self, X_train, y_train, X_test, y_test):
        # Chiamato per preparare il modello difeso
        self.images = X_test
        self.true_targets = y_test
        self.num_samples = X_test.shape[0]
        self.train(X_train, y_train)
        print("Model training finished.")
        self.test(X_test, y_test)
        print("Model testing finished. Initial accuracy score: " + str(self.initial_score))

    def defense_attack(self, attack_method, epsilon):
        # Assicurati che il costruttore di Attack venga chiamato correttamente
        super().create_one_hot_targets(self.true_targets)
        perturbed_images, highest_epsilon = self.perturb_images(epsilon, attack_method)

        # Applica la tua strategia di difesa
        defended_images = self.defense_method(perturbed_images)

        defended_preds = self.model.predict(defended_images)
        score = accuracy_score(self.true_targets, defended_preds)
        return defended_images, defended_preds, score, highest_epsilon

    def defense_attack_to_max_epsilon(self, attack_method, max_epsilon):
        # Assicurati che il costruttore di Attack venga chiamato correttamente
        super().create_one_hot_targets(self.true_targets)

        self.max_epsilon = max_epsilon
        self.scores = []
        self.epsilons = []
        self.defended_images_per_epsilon = []
        self.defended_outputs_per_epsilon = []
        for epsilon in range(0, self.max_epsilon):
            defended_images, defended_preds, score, highest_epsilon = self.defense_attack(attack_method, epsilon)
            self.epsilons.append(highest_epsilon)
            self.scores.append(score)
            self.defended_images_per_epsilon.append(defended_images)
            self.defended_outputs_per_epsilon.append(defended_preds)
# Carica il modello
model = joblib.load('model/modello.pkl')

# Creazione dell'oggetto DefenseAttack
defense_attack = DefenseAttack(model)


# Esegui la difesa con PCA
defense_attack.prepare_defense(X_train, y_train, X_test, y_test)
defended_images, _, _, _ = defense_attack.defense_attack(targeted_gradient, 30)

# Scegli un'immagine prima della difesa e dopo la difesa
original_image = X_test[0]
defended_image = defended_images[0]

# Visualizza l'immagine originale
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_image.reshape(28, 28), cmap='gray')
plt.title('Immagine originale')

# Visualizza l'immagine difesa
plt.subplot(1, 2, 2)
plt.imshow(defended_image.reshape(28, 28), cmap='gray')
plt.title('Immagine difesa con PCA')

plt.show()

# Attacco con epsilon massimo
defense_attack.defense_attack_to_max_epsilon(targeted_gradient, 30)

# Calcola le metriche di prestazione dopo la difesa solo se la difesa è stata effettuata con successo
if hasattr(defense_attack, 'defended_outputs_per_epsilon'):
    defended_preds = defense_attack.defended_outputs_per_epsilon[-1]
    defended_accuracy = accuracy_score(y_test, defended_preds)
    print("Accuracy after Defense:", defended_accuracy)
else:
    print("La difesa non è stata effettuata. Controlla la logica della tua difesa.")

# Visualizza i risultati della difesa solo se la difesa è stata effettuata con successo
if hasattr(defense_attack, 'epsilons') and hasattr(defense_attack, 'scores'):
    plt.figure(figsize=(10, 5))
    plt.plot(defense_attack.epsilons, defense_attack.scores, 'g*')
    plt.ylabel('accuracy_score')
    plt.xlabel('epsilon')
    plt.title('Accuracy score breakdown - Defense against targeted attack')
    plt.show()
else:
    print("La visualizzazione della difesa non è possibile. Controlla la logica della tua difesa.")

# Calcola l'accuracy dell'attaccante dopo la difesa
if hasattr(defense_attack, 'defended_outputs_per_epsilon'):
    # Ottieni le immagini difese
    defended_images = defense_attack.defended_images_per_epsilon[-1]

    # Calcola le predizioni dell'attaccante sulle immagini difese
    _, attacker_preds, _, _ = defense_attack.attack(targeted_gradient, 30)

    # Calcola l'accuracy dell'attaccante
    attacker_accuracy = accuracy_score(y_test, attacker_preds)
    print("Accuracy dell'attaccante dopo la difesa:", attacker_accuracy)
else:
    print("La difesa non è stata effettuata. Controlla la logica della tua difesa.")