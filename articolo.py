
from subprocess import check_output

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset/train.csv")
print(df.head())

y = df.label.values
X = df.drop("label",axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

fig1, ax1 = plt.subplots(1,15, figsize=(15,10))

for i in range(15):
    ax1[i].imshow(X_test[i].reshape((28,28)), cmap="gray_r")
    ax1[i].axis('off')
    ax1[i].set_title(y_test[i])
plt.show()

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

def non_targeted_gradient(target, pred_proba, weights):
    # Calcola la differenza tra le probabilità predette e i target reali
    gradient = pred_proba - target
    # Calcola il gradiente non mirato
    gradient = np.dot(weights.T, gradient.T).T
    return gradient


model = LogisticRegression(multi_class='multinomial',
                           solver='lbfgs',
                           fit_intercept=False,
                           max_iter=1000)


attack = Attack(model)
attack.prepare(X_train, y_train, X_test, y_test)

weights = attack.weights
print(weights.shape)

num_classes = len(np.unique(y_train))
print(num_classes)

"""
Non-targeted Attack: Maximizing output-target discrepance 
"""
# Attacco non mirato
attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_gradient, 30)
non_targeted_scores = attack.scores
print(attack.scores)

sns.set()
plt.figure(figsize=(10,5))
plt.plot(attack.epsilons, attack.scores, 'g*')
plt.ylabel('accuracy_score')
plt.xlabel('epsilon')
plt.title('Accuracy score breakdown - non-targeted attack');
plt.show()

eps = 16
print(attack.epsilons[eps])

example_images = attack.perturbed_images_per_epsilon[eps]
example_preds = attack.perturbed_outputs_per_epsilon[eps]

example_results = pd.DataFrame(data=attack.true_targets, columns=['y_true'])
example_results['y_fooled'] = example_preds
example_results['y_predicted'] = attack.preds
example_results['id'] = example_results.index.values
print(example_results.head())

success_df = example_results[example_results.y_fooled != example_results.y_true]
print(success_df.head())

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
        ax2[i,j].set_title("true: " + str(y_test[example_id]) + ", fooled: " + str(y_fooled)  + "\n"
                           + "epsilon: " + str(int(epsilon)))
plt.show()

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15,5))
axB.imshow(example_images[example_id].reshape((28,28)), cmap='Greens')
axB.set_title("Non-targeted attack result: " + str(example_preds[example_id]))
axA.imshow(X_test[example_id].reshape((28,28)), cmap='Greens')
axA.set_title("True label: " + str(y_test[example_id]))
axC.imshow((X_test[example_id]-example_images[example_id]).reshape((28,28)), cmap='Reds')
axC.set_title("Perturbation: epsilon 16")

plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='y_fooled', data=example_results[example_results.y_true != example_results.y_fooled])
plt.show()

wrong_predictions = example_results[example_results.y_true != example_results.y_predicted]
print(wrong_predictions.shape)

plt.figure(figsize=(10,5))
sns.countplot(x='y_predicted', data=wrong_predictions)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='y_true', data=wrong_predictions)
plt.show()

attacktargets = example_results.loc[example_results.y_true != example_results.y_fooled].groupby(
    'y_true').y_fooled.value_counts()
counts = example_results.loc[example_results.y_true != example_results.y_fooled].groupby(
    'y_true').y_fooled.count()
attacktargets = attacktargets/counts * 100
attacktargets = attacktargets.unstack()
attacktargets = attacktargets.fillna(0.0)
attacktargets = attacktargets.apply(np.round).astype(int)

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(attacktargets, annot=True, ax=ax, cbar=False, square=True, cmap="Reds", fmt="g")
ax.set_title("How often was y_true predicted as some y_fooled digit in percent?")
plt.show()

"""
One example image

"""
example = X_test[0]
plt.imshow(example.reshape((28,28)), cmap='Greens')
print("true label target: " + str(y_test[0]))

fooling_classes = []
for k in range(num_classes):
    if k != y_test[0]:
        fooling_classes.append(k)
print(fooling_classes)

foolingtargets = np.zeros((len(fooling_classes), num_classes))
for n in range(len(fooling_classes)):
    foolingtargets[n,fooling_classes[n]] = 1
print(foolingtargets)


"""
attack the model
"""
def targeted_gradient(target, pred_proba, weights):
    # Calcola la differenza tra i target desiderati e le probabilità predette
    gradient = target - pred_proba
    # Calcola il gradiente mirato
    gradient = np.dot(weights.T, gradient.T).T
    return gradient

eps=100
targeted_perturbed_images = []
targeted_perturbed_predictions = []
for fooling_target in foolingtargets:
    targeted_perturbation = attack.get_perturbation(eps, targeted_gradient, fooling_target, attack.preds_proba[0])
    targeted_perturbed_image = X_test[0] + targeted_perturbation
    targeted_perturbed_prediction = attack.model.predict(targeted_perturbed_image.reshape(1, -1))
    targeted_perturbed_images.append(targeted_perturbed_image)
    targeted_perturbed_predictions.append(targeted_perturbed_prediction)

print(targeted_perturbed_predictions)


fig3, ax3 = plt.subplots(3,3, figsize=(9,9))
for i in range(3):
    for j in range(3):
        ax3[i,j].imshow(targeted_perturbed_images[i*3+j].reshape((28,28)), cmap="Greens")
        ax3[i,j].axis('off')
        ax3[i,j].set_title("fooling result: " + str(targeted_perturbed_predictions[i*3+j][0]))

plt.show()

"""
Natural vs. non-natural targeted attack
"""

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(attacktargets, annot=True, ax=ax, cbar=False, cmap="Purples", fmt="g");
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
plt.title('Accuracy score breakdown: natural vs non-natural targeted attack')
plt.show()


"""
Comparison to "discrepance" Gradient Sign Method¶

"""
def non_targeted_sign_gradient(target, pred_proba, weights):
    # Calcola la differenza tra le probabilità predette e i target reali
    gradient = pred_proba - target
    # Calcola il gradiente non mirato
    gradient = np.dot(weights.T, gradient.T).T
    # Prendi il segno del gradiente
    sign_gradient = np.sign(gradient)
    return sign_gradient

attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_sign_gradient, 30)

plt.figure(figsize=(10,5))
gm, = plt.plot(attack.epsilons, non_targeted_scores, 'g*', label='gradient method')
gsm, = plt.plot(attack.epsilons, attack.scores, 'r*', label='gradient sign method')
plt.ylabel('accuracy_score')
plt.xlabel('eta')
plt.legend(handles=[gm, gsm])
plt.title('Accuracy score breakdown')
plt.show()

# Esecuzione del metodo del segno del gradiente
attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_sign_gradient, 30)
gsm_scores = attack.scores

# Visualizzazione dei risultati
plt.figure(figsize=(10, 5))
plt.plot(attack.epsilons, non_targeted_scores, 'g*', label='Non-targeted Gradient Method')
plt.plot(attack.epsilons, natural_scores, 'b*', label='Natural Targeted Attack')
plt.plot(attack.epsilons, non_natural_scores, 'r*', label='Non-natural Targeted Attack')
plt.plot(attack.epsilons, gsm_scores, 'c*', label='Gradient Sign Method')
plt.ylabel('Accuracy Score')
plt.xlabel('Epsilon')
plt.legend()
plt.title('Accuracy Score Comparison of Different Attacks')
plt.show()
