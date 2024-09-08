import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

# Definisci l'architettura dell'autoencoder
input_img = Input(shape=(28, 28, 1))  # Le immagini MNIST sono 28x28 in scala di grigi

# Encoder
x = Flatten()(input_img)
encoded = Dense(64, activation='relu')(x)

# Bottleneck
bottleneck = Dense(32, activation='relu')(encoded)

# Decoder
x = Dense(64, activation='relu')(bottleneck)
decoded = Dense(28 * 28, activation='sigmoid')(x)
decoded = Reshape((28, 28, 1))(decoded)

# Crea il modello autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Ignora i warning di convergenza
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Carica il dataset
df = pd.read_csv("dataset/train.csv")
print(df.head())

# Prepara i dati
y = df.label.values
X = df.drop("label", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Standardizza i dati per evitare warning nel modello SVC
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Visualizza alcune immagini di test
fig1, ax1 = plt.subplots(1, 15, figsize=(15, 10))
for i in range(15):
    ax1[i].imshow(X_test[i].reshape((28, 28)), cmap="gray_r")
    ax1[i].axis('off')
    ax1[i].set_title(y_test[i])
plt.show()


# Classe per l'attacco
class Attack:

    def __init__(self, model):
        self.fooling_targets = None
        self.model = model

    def prepare(self, X_train, y_train, X_test, y_test):
        self.images = X_test
        self.true_targets = y_test
        self.num_samples = X_test.shape[0]
        self.train(X_train, y_train)
        print("Addestramento del modello completato.")
        self.test(X_test, y_test)
        print("Test del modello completato. Punteggio di accuratezza iniziale: " + str(self.initial_score))

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
        if inf_norm == 0:
            inf_norm = 1  # Evita la divisione per zero
        perturbation = epsilon / inf_norm * gradient
        perturbation = np.clip(perturbation, -epsilon, epsilon)  # Limita la perturbazione
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


# Funzione per il gradiente non mirato
def non_targeted_gradient(target, pred_proba, weights):
    gradient = pred_proba - target
    gradient = np.dot(weights.T, gradient.T).T
    return gradient


# Funzione per il gradiente del segno non mirato
def non_targeted_sign_gradient(target, pred_proba, weights):
    gradient = pred_proba - target
    gradient = np.dot(weights.T, gradient.T).T
    sign_gradient = np.sign(gradient)
    return sign_gradient


# Funzione per il gradiente mirato
def targeted_gradient(target, pred_proba, weights):
    gradient = target - pred_proba
    gradient = np.dot(weights.T, gradient.T).T
    return gradient


# Modello di regressione logistica
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False, max_iter=5000)

# Creazione dell'attacco
attack = Attack(model)
attack.prepare(X_train, y_train, X_test, y_test)

# Valutazione del modello prima dell'attacco
initial_accuracy = accuracy_score(y_test, attack.preds)
print(f"Accuratezza iniziale del modello: {initial_accuracy}")

# Inizializza i target one-hot
attack.create_one_hot_targets(y_test)


# Feature Squeezing
def feature_squeezing(X, bit_depth=8):
    X_squeezed = np.floor(X * (2 ** bit_depth)) / (2 ** bit_depth)
    return X_squeezed


# Applicazione del feature squeezing
X_test_squeezed = feature_squeezing(X_test)

# Creazione del dataset di rilevamento
X_detection = np.vstack([X_test, X_test_squeezed])
X_detection_squeezed = np.vstack([X_test_squeezed, feature_squeezing(X_test_squeezed)])
y_detection = np.hstack([np.zeros(len(X_test)), np.ones(len(X_test_squeezed))])



# Addestramento del rilevatore
ensemble_detector = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(max_depth=2)),
    ('svc', SVC(probability=True, kernel='linear', max_iter=200)),
    ('rf', RandomForestClassifier(n_estimators=50))
], voting='soft')

X_train_detect, X_test_detect, y_train_detect, y_test_detect = train_test_split(X_detection_squeezed, y_detection,
                                                                                test_size=0.4, random_state=0)
ensemble_detector.fit(X_train_detect, y_train_detect)

# Prepara i dati per l'autoencoder
X_train_auto = X_train.reshape(-1, 28, 28, 1)  # Modella i dati in un formato adeguato per l'autoencoder
X_test_auto = X_test.reshape(-1, 28, 28, 1)

# Normalizza i dati
X_train_auto = X_train_auto.astype('float32') / 255.
X_test_auto = X_test_auto.astype('float32') / 255.

# Addestra l'autoencoder
autoencoder.fit(X_train_auto, X_train_auto, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_auto, X_test_auto))


# Funzione per la classificazione con il rilevatore e autoencoder
def classify_with_autoencoder(model, X, detector, autoencoder, threshold=0.5):
    X_squeezed = feature_squeezing(X)
    detection_probs = detector.predict_proba(X_squeezed)[:, 1]

    adversarial_indices = detection_probs >= threshold
    benign_indices = detection_probs < threshold

    if np.any(adversarial_indices):
        X_adversarial = X[adversarial_indices].reshape(-1, 28, 28, 1).astype('float32') / 255.
        X_adversarial_reconstructed = autoencoder.predict(X_adversarial)
        X_adversarial_reconstructed = X_adversarial_reconstructed.reshape(-1, 28 * 28)
        corrected_preds = model.predict(X_adversarial_reconstructed)
    else:
        corrected_preds = []

    if np.any(benign_indices):
        X_benign = X[benign_indices]
        benign_preds = model.predict(X_benign)
    else:
        benign_preds = []

    preds = np.zeros(X.shape[0])
    preds[benign_indices] = benign_preds
    preds[adversarial_indices] = corrected_preds

    return preds



# Funzione per eseguire l'attacco e confrontare le accuratezze con autoencoder
def test_and_compare_attack_with_autoencoder(attack_method, attack_name, epsilons, autoencoder):
    scores_no_detector = []
    scores_with_autoencoder = []
    final_preds = None  # Variabile per memorizzare le predizioni finali

    for epsilon in epsilons:
        perturbed_images, perturbed_preds, score, _ = attack.attack(attack_method, epsilon)
        scores_no_detector.append(score)

        final_preds = classify_with_autoencoder(model, X_test, ensemble_detector, autoencoder)
        final_accuracy = accuracy_score(y_test, final_preds)
        scores_with_autoencoder.append(final_accuracy)

        print(f"Accuratezza senza rilevatore dopo l'attacco '{attack_name}' con epsilon={epsilon}: {score}")
        print(f"Accuratezza con autoencoder dopo l'attacco '{attack_name}' con epsilon={epsilon}: {final_accuracy}")

    return scores_no_detector, scores_with_autoencoder, final_preds

# Creazione dei target per attacchi mirati naturali e non-naturali
# Assicurarsi che la colonna y_fooled sia correttamente definita e non contenga stringhe
attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_gradient, 30)

# Creazione del dizionario dei target
natural_targets_dict = {}
non_natural_targets_dict = {}

# Assegna gli indici dei target mirati
for ix in range(attack.num_classes):
    # Ottiene le predizioni per ogni classe
    series = pd.Series(attack.scores)
    natural_targets_dict[ix] = series.idxmax()  # Target naturale
    non_natural_targets_dict[ix] = series.idxmin()  # Target non naturale

# Assegna i target di "fooling" naturali e non naturali
natural_foolingtargets = np.zeros((y_test.shape[0]))
non_natural_foolingtargets = np.zeros((y_test.shape[0]))

for n in range(len(natural_foolingtargets)):
    target = y_test[n]
    natural_foolingtargets[n] = natural_targets_dict[int(target)]
    non_natural_foolingtargets[n] = non_natural_targets_dict[int(target)]

# Ora è possibile eseguire gli attacchi mirati


# Attacchi da testare
epsilons = [0.01, 0.05, 0.1, 5, 10]
attacks = [
    (non_targeted_gradient, "Gradiente Non Mirato"),
    (non_targeted_sign_gradient, "Gradiente del Segno Non Mirato"),
    (lambda target, pred_proba, weights: targeted_gradient(target, pred_proba, weights),
     "Attacco Mirato Naturale"),
    (lambda target, pred_proba, weights: targeted_gradient(target, pred_proba, weights),
     "Attacco Mirato Non-Naturale")
]

# Esegui gli attacchi e confronta le accuratezze
results = {}
final_preds = None
for attack_method, attack_name in attacks:
    if "Naturale" in attack_name:
        if "Non-Naturale" in attack_name:
            attack.create_one_hot_targets(non_natural_foolingtargets.astype(int))
        else:
            attack.create_one_hot_targets(natural_foolingtargets.astype(int))
    scores_no_detector, scores_with_detector, final_preds = test_and_compare_attack_with_autoencoder(attack_method, attack_name, epsilons, autoencoder)
    results[attack_name] = (scores_no_detector, scores_with_detector)


# Visualizza i risultati per ogni attacco
for attack_name in results:
    scores_no_detector, scores_with_detector = results[attack_name]
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, scores_no_detector, 'r*-', label=f'{attack_name} senza rilevatore')
    plt.plot(epsilons, scores_with_detector, 'g*-', label=f'{attack_name} con rilevatore')
    plt.xlabel('Epsilon')
    plt.ylabel('Punteggio di Accuratezza')
    plt.title(f'Confronto delle Accuratezze per {attack_name}')
    plt.legend()
    plt.show()

# Stampa l'accuratezza finale del modello con il rilevatore dopo tutti gli attacchi
if final_preds is not None:
    final_accuracy = accuracy_score(y_test, final_preds)
    print(f"Accuratezza finale del modello con rilevatore dopo tutti gli attacchi: {final_accuracy}")

