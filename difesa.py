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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("dataset/train.csv")
print(df.head())
"""
 l'etichetta contiene il vero numero e le altre colonne contengono tutti i 784 pixel di un'immagine con una 
 dimensione di 28 per 28 pixel. Dividiamo i nostri dati in un set di addestramento e un set di test. 
 In questo modo possiamo valutare le prestazioni del nostro modello sul set di test e vedere come questo punteggio si comporta durante l'attacco.
"""
y = df.label.values
X = df.drop("label",axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape)

# Prima di iniziare a creare attacchi mirati e non mirati, diamo un'occhiata ai primi numeri del set di test:
fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    ax1[i].imshow(X_test[i].reshape((28,28)), cmap="gray_r")
    ax1[i].axis('off')
    ax1[i].set_title(y_test[i])
plt.show()
import tensorflow as tf
from tensorflow import keras
#Ho scritto una piccola classe che esegue l'attacco alla regressione logistica
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

def calc_output_weighted_weights(output, w):
    for c in range(len(output)):
        if c == 0:
            weighted_weights = output[c] * w[c]
        else:
            weighted_weights += output[c] * w[c]
    return weighted_weights

def targeted_gradient(foolingtarget, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(output)):
        if k == 0:
            gradient = foolingtarget[k] * (w[k]-ww)
        else:
            gradient += foolingtarget[k] * (w[k]-ww)
    return gradient

def non_targeted_gradient(target, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(target)):
        if k == 0:
            gradient = (1-target[k]) * (w[k]-ww)
        else:
            gradient += (1-target[k]) * (w[k]-ww)
    return gradient

def non_targeted_sign_gradient(target, output, w):
    gradient = non_targeted_gradient(target, output, w)
    return np.sign(gradient)
# Crea un modello Sequential di TensorFlow

#Prima di tutto, abbiamo bisogno di un modello per la regressione logistica multiclasse:
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)
"""

E passeremo questo al nostro oggetto e chiameremo il metodo "prepare". In questo modo addestreremo il nostro modello sui dati di addestramento 
e otterremo il punteggio di accuratezza iniziale sui dati di test. In seguito vogliamo analizzare questo punteggio alterando i dati di test
"""
attack = Attack(model)
attack.prepare(X_train, y_train, X_test, y_test)



# Esegui l'attacco mirato con epsilon 30
#targeted_attack = attack.run_attack("targeted", 30)

"""
Okay, circa il 90 percento dei dati di test è stato classificato correttamente e per noi questo è sufficiente per iniziare a giocare. 
Abbiamo presumuto che gli input e gli appartenenti alle classi siano indipendenti e identicamente distribuiti.
 L'obiettivo tn di un input xn è un vettore con K elementi che segue la codifica one-hot (la vera classe etichetta è 1, tutte le altre sono 0).
  Massimizzare la probabilità di corrispondenze sopra è anche chiamato approccio della massima verosimiglianza. 
  Ogni classe in una regressione logistica multiclasse ha il proprio vettore di pesi e gli input vengono passati attraverso la funzione softmax con i pesi per ottenere
   l'output del modello.
"""
weights = attack.weights
print(weights.shape)

num_classes = len(np.unique(y_train))
print(num_classes)

"""
Attacco non mirato: Massimizzazione della discrepanza tra output e target

In analogia all'approccio della massima verosimiglianza, definisco una funzione di discrepanza come obiettivo. 
Di solito, la verosimiglianza ci dà la probabilità che l'output del nostro modello \(y\) corrisponda al target \(t\). 
Quindi, massimizzare la verosimiglianza ci fornisce le migliori corrispondenze. 
Creando un attacco, dobbiamo pensare in modo opposto: vogliamo massimizzare la probabilità che gli output non corrispondano ai target. 
Nota che, a differenza della funzione di verosimiglianza, ora abbiamo \(1-t\) invece di \(t\). 

Per addestrare un modello, di solito massimizziamo la verosimiglianza rispetto ai parametri dei pesi mentre gli input sono fissi.
 Nel nostro caso, abbiamo già un modello addestrato e pesi fissi. Ma possiamo aggiungere piccole perturbazioni alle nostre immagini 
 di input in modo da massimizzare la nostra funzione di discrepanza. Facciamo ciò e semplifichiamo le cose utilizzando il logaritmo! :-)

 dovremmo tenere presente che il gradiente delle attivazioni produce il vettore di pesi della classe:

Cerchiamo di capire questo: per l'etichetta vera, il sommand è 0, mentre tutte le altre classi contribuiscono al gradiente con il loro vettore di pesi della classe ridotto da una somma "ponderata" degli altri pesi delle classi. Cosa significa questo?...

Per massimizzare la discrepanza per giocare, possiamo utilizzare la discesa del gradiente. Tuttavia, dobbiamo trovare un tasso sufficiente \(\eta\). Dato un input \(x_m\), aggiungeremo quindi una perturbazione

Innanzitutto, dobbiamo calcolare le perturbazioni per ogni immagine nel set di test.
 Per farlo, dobbiamo trasformare i nostri veri target in one-hot-targets e chiamare l'attacco :-). 
 Poiché voglio vedere quanto epsilon è necessario per creare uno scomposizione efficace, utilizzo il metodo attack_to_max_epsilon.
"""

attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_gradient, 30)
non_targeted_scores = attack.scores

sns.set()
plt.figure(figsize=(10,5))
plt.plot(attack.epsilons, attack.scores, 'g*')
plt.ylabel('accuracy_score')
plt.xlabel('epsilon')
plt.title('Accuracy score breakdown - non-targeted attack')

plt.show()

"""
la soglia è data da un massimo di 16 pixel che possono essere aggiunti come perturbazione per ogni pixel per immagine. Con questo  ϵ,
 finiremmo con un modello che prevede ancora circa il 40% correttamente. Se usassimo un massimo di ϵ=30, il modello fallirebbe con quasi il 90% dei numeri nel set di test :-) .
  Diamo un'occhiata a un esempio di inganno riuscito per una serie di ϵ fino a un massimo di ϵ=16.
"""

eps = 16
print(attack.epsilons[eps])

#Abbiamo bisogno delle immagini perturbate così come dei risultati di inganno per quel ϵ

example_images = attack.perturbed_images_per_epsilon[eps]
example_preds = attack.perturbed_outputs_per_epsilon[eps]

# E salverò i risultati in un dataframe di pandas in modo che possiamo trovare facilmente gli inganni riusciti:

example_results = pd.DataFrame(data=attack.true_targets, columns=['y_true'])
example_results['y_fooled'] = example_preds
example_results['y_predicted'] = attack.preds
example_results['id'] = example_results.index.values
print(example_results.head())

success_df = example_results[example_results.y_fooled != example_results.y_true]
print(success_df.head())

# Okay, sceglieremo uno di questi esempi riusciti e traceremo la relativa immagine perturbata su una serie di  ϵ:
example_id = success_df.id.values[0]
print(example_id)

fig2, ax2 = plt.subplots(4,4, figsize=(15,15))
for i in range(4):
    for j in range(4):
        image = attack.perturbed_images_per_epsilon[i*4 + j][example_id]
        y_fooled = attack.perturbed_outputs_per_epsilon[i*4 + j][example_id]
        epsilon = attack.epsilons[i*4 +j]
        ax2[i,j].imshow(image.reshape((28,28)), cmap="gray_r")
        ax2[i,j].axis('off')
        ax2[i,j].set_title("true: " + str(y_test[example_id]) + ", fooled: " + str(y_fooled)  + "\n" + "epsilon: " + str(int(epsilon)))

plt.show()

"""
Sì! :-) Possiamo ancora vedere il vero target e non il target fuorviato. 
È incredibile. Ma possiamo anche notare che lo sfondo ha intensità aumentata.
 Visualizziamo la differenza tra l'etichetta vera originale e l'immagine avversaria per ϵ=16:
"""

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15,5))
axB.imshow(example_images[example_id].reshape((28,28)), cmap='Greens')
axB.set_title("Non-targeted attack result: " + str(example_preds[example_id]))
axA.imshow(X_test[example_id].reshape((28,28)), cmap='Greens')
axA.set_title("True label: " + str(y_test[example_id]))
axC.imshow((X_test[example_id]-example_images[example_id]).reshape((28,28)), cmap='Reds')
axC.set_title("Perturbation: epsilon 16");
plt.show()

"""
Il percorso guida del gradiente - obiettivi naturali di inganno
Sono felice che sia stato possibile ingannare il nostro modello, ma è ancora diffuso e poco chiaro dove ci guidi il gradiente in un solo passo 
(ricorda che non iteriamo con la discesa del gradiente, facciamo solo un passo e la dimensione è data dalla forza del gradiente moltiplicata per η).
 Assumo che alcuni numeri siano più vicini tra loro nello spazio dei pesi rispetto ad altri.
 Poiché l'addestramento del modello traccia i confini decisionali in base alla qualità dei dati di input e alla flessibilità dell'architettura del modello, ci saranno regioni in cui un 3 non viene previsto come 3 ma come 8. Queste sono le regioni in cui il modello fa una previsione incorretta. E penso che ci siano numeri preferiti come previsioni errate date un'immagine di input di un certo numero. Forse i gradienti di inganno ci guidano verso quei numeri "naturali" di inganno?
"""

plt.figure(figsize=(10,5))
sns.countplot(x='y_fooled', data=example_results[example_results.y_true != example_results.y_fooled])
plt.show()

"""

OK, vediamo che l'8 è stato selezionato più spesso come obiettivo di inganno. 
Ma anche il 9, il 3, il 5 e il 2 hanno conteggi elevati, a differenza di 0, 1, 6 e 7. 
Se la nostra ipotesi è corretta, ovvero che il gradiente ci conduca a obiettivi in cui il modello tende a fallire nella previsione,
 dovremmo vedere un modello simile di conteggi per previsioni errate:
"""
wrong_predictions = example_results[example_results.y_true != example_results.y_predicted]
print(wrong_predictions.shape)

print(X_test.shape)

"""
OK, quindi su 16800 campioni, il modello non è riuscito a prevedere circa 1600 volte.
 Ecco perché il nostro punteggio di accuratezza iniziale è vicino al 90% (cioè il 10% di errori).
  Ora, quale cifra è stata selezionata come risultato di previsione errata più spesso?
"""

plt.figure(figsize=(10,5))
sns.countplot(x='y_predicted', data=wrong_predictions)
plt.show()

"""
Sì, è lo stesso modello dei target di inganno. 
Poiché ciò è causato dalla difficoltà del nostro modello nel tracciare buoni confini decisionali,
 dovremmo vedere questo modello anche per le etichette reali di quei numeri che sono stati previsti in modo errato:
"""

plt.figure(figsize=(10,5))
sns.countplot(x='y_true', data=wrong_predictions)
plt.show()
"""
Ora voglio vederlo in modo più dettagliato: Quali sono i bersagli naturali di inganno (per inganni riusciti) per ciascuna cifra?
"""
attacktargets = example_results.loc[example_results.y_true != example_results.y_fooled].groupby(
    'y_true').y_fooled.value_counts()
counts = example_results.loc[example_results.y_true != example_results.y_fooled].groupby(
    'y_true').y_fooled.count()
attacktargets = attacktargets/counts * 100
attacktargets = attacktargets.unstack()
attacktargets = attacktargets.fillna(0.0)
attacktargets = attacktargets.apply(np.round).astype(int)

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(attacktargets, annot=True, ax=ax, cbar=False, square=True, cmap="Reds", fmt="g");
ax.set_title("How often was y_true predicted as some y_fooled digit in percent?");
plt.show()

"""

Abbiamo scoperto che ogni cifra ha il suo obiettivo naturale di inganno. 
Ad esempio, è probabile che il 3 sia previsto come 5 o 8, il che ha senso poiché tutti e tre hanno "due bolle" 
una sopra l'altra come forma di schizzo. Puoi vedere che l'8 è stata una buona scelta per diversi numeri...
 interessantemente anche per l'1. Inoltre, troviamo che per alcune immagini vale entrambe le cose: ad esempio, 4 e 9 sono possibili in entrambi i modi.
"""

"""
Attacco mirato
Abbiamo visto che ingannare il modello di regressione logistica multiclasse è stato facile
 con la discesa del gradiente e l'unica parte complicata era calcolare il gradiente rispetto 
 agli input della nostra funzione di discrepanza. Invece di forzare la funzione a restituire la massima discrepanza,
  avremmo potuto anche costruirla in modo che gli output debbano corrispondere a un obiettivo falso specifico.
  Per sperimentare, selezioniamo un input da Xtest e cerchiamo di effettuare attacchi mirati per ogni classe , tranne per il vero obiettivo di etichetta 
"""
example = X_test[0]
imshow(example.reshape((28,28)), cmap='Greens')
print("true label target: " + str(y_test[0]))

"""
Innanzitutto, abbiamo bisogno di alcuni obiettivi di inganno. Per il nostro esempio di cifra, tutte le altre cifre sono possibili:
"""
fooling_classes = []
for k in range(num_classes):
    if k != y_test[1]:
        fooling_classes.append(k)
print(fooling_classes)

foolingtargets = np.zeros((len(fooling_classes), num_classes))
for n in range(len(fooling_classes)):
    foolingtargets[n,fooling_classes[n]] = 1

"""
Forzerò l'attacco per avere successo consentendo un epsilon sufficientemente alto per ottenere tutti gli obiettivi. 
In questo modo possiamo comunque scoprire se possiamo vedere l'etichetta vera o l'obiettivo di inganno.
"""

eps=100
targeted_perturbed_images = []
targeted_perturbed_predictions = []
for fooling_target in foolingtargets:
    targeted_perturbation = attack.get_perturbation(eps, targeted_gradient, fooling_target, attack.preds_proba[0])
    targeted_perturbed_image = X_test[0] + targeted_perturbation
    targeted_perturbed_prediction = attack.model.predict(targeted_perturbed_image.reshape(1, -1))
    targeted_perturbed_images.append(targeted_perturbed_image)
    targeted_perturbed_predictions.append(targeted_perturbed_prediction)

fig3, ax3 = plt.subplots(3,3, figsize=(9,9))
for i in range(3):
    for j in range(3):
        ax3[i,j].imshow(targeted_perturbed_images[i*3+j].reshape((28,28)), cmap="Greens")
        ax3[i,j].axis('off')
        ax3[i,j].set_title("fooling result: " + str(targeted_perturbed_predictions[i*3+j][0]))


plt.show()

"""

Anche se possiamo vedere un forte rumore di fondo, l'etichetta vera non è distrutta.
 Posso ancora vedere l'etichetta vera con i miei occhi, mentre il modello prevede l'obiettivo di inganno desiderato
  (0 a 9, tranne l'etichetta vera
"""

"""

Ora mi piacerebbe vedere cosa succede con il punteggio di accuratezza se inganniamo il modello per ogni immagine nel set di test.
 Analizzando gli attacchi non mirati, abbiamo scoperto che alcuni numeri vengono più utilizzati come "obiettivi" di inganno rispetto
  ad altri e che ogni cifra ha la sua controparte come cifra di inganno. Presumo che l'inganno avvenga nelle regioni in cui il modello 
  non riesce a tracciare buoni confini decisionali. Utilizzando attacchi mirati dovremmo vedere che possiamo abbattere più facilmente
   il punteggio di accuratezza con obiettivi naturali di inganno rispetto agli altri numeri. Proviamolo!
   l conteggio più alto rappresenta l'obiettivo naturale di inganno, mentre il più basso corrisponde all'obiettivo di inganno non naturale. Date le heatmap,
    potremmo creare gli obiettivi utilizzando argmin e argmax per riga (y_true) nel seguente modo:
"""
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(attacktargets, annot=True, ax=ax, cbar=False, cmap="Purples", fmt="g")
plt.show()

natural_targets_dict = {}
non_natural_targets_dict = {}
for ix, series in attacktargets.iterrows():
    natural_targets_dict[ix] = series.argmax()
    non_natural_targets_dict[ix] = series.drop(ix).argmin()

print(natural_targets_dict)

natural_foolingtargets = np.zeros((y_test.shape[0]))
non_natural_foolingtargets = np.zeros((y_test.shape[0]))

for n in range(len(natural_foolingtargets)):
    target = y_test[n]
    natural_foolingtargets[n] = natural_targets_dict[target]
    non_natural_foolingtargets[n] = non_natural_targets_dict[target]

attack.create_one_hot_targets(natural_foolingtargets.astype(int))
attack.attack_to_max_epsilon(targeted_gradient, 30)
natural_scores = attack.scores
attack.create_one_hot_targets(non_natural_foolingtargets.astype(int))
attack.attack_to_max_epsilon(targeted_gradient, 30)
non_natural_scores = attack.scores

plt.figure(figsize=(10,5))
nf, = plt.plot(attack.epsilons, natural_scores, 'g*', label='natural fooling')
nnf, = plt.plot(attack.epsilons, non_natural_scores, 'b*', label='non-natural fooling')
plt.legend(handles=[nf, nnf])
plt.ylabel('accuracy_score')
plt.xlabel('epsilon')
plt.title('Accuracy score breakdown: natural vs non-natural targeted attack');
plt.show()

"""
Possiamo vedere chiaramente che è stato più facile ingannare il modello con obiettivi di inganno naturali.
"""

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

# Creazione dell'oggetto DefenseAttack
defense_attack = DefenseAttack(model)

# Preparazione del modello difeso
defense_attack.prepare_defense(X_train, y_train, X_test, y_test)

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