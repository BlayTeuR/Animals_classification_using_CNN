# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 2 
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.keras"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath         
# - number_images        
# - number_images_class_x
# - image_scale          
# - images_color_mode    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "donnees/"
testPath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images_class_0 = 1000   # baleine
number_images_class_1 = 1000   # requin
number_images_class_2 = 1000   # requin-baleine

# La taille des images à classer
image_scale = 128

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,
    batch_size=1,
    color_mode=images_color_mode)

# plus besoin de faire cela -> (x, y_true) = test_itr.__next__()

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0]*number_images_class_0 +
                  [1]*number_images_class_1 +
                  [2]*number_images_class_2)


# evaluation du modele
#test_eval = Classifier.evaluate_generator(test_itr, verbose=1)
test_eval = Classifier.evaluate(test_itr, verbose=1)
# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
pred_proba = Classifier.predict(test_itr, verbose=1)
predicted_classes = np.argmax(pred_proba, axis=1)
# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************

# 1) Création de la matrice de confusion
cm = confusion_matrix(y_true, predicted_classes)
print(cm)

plt.imshow(cm, cmap="Blues")
plt.title("Matrice de confusion")
plt.colorbar()
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

# Téléchargement de la matrice de confusion
fig = plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Matrice de confusion")
plt.colorbar()
plt.xlabel("Prédit")
plt.ylabel("Réel")

fig.savefig("matrice_confusion.png", dpi=300, bbox_inches='tight')
plt.savefig("matrice_confusion.png")

plt.show()
plt.close(fig)

# 2)  Extraction des erreurs

class_names = ["baleine", "requin", "requinbaleine"]
errors = np.where(predicted_classes != y_true)[0]

print("\nNombre total d'images mal classées :", len(errors))

# Stockage des images mal classés

sample_errors = {}
for i in errors:
    real = y_true[i]
    pred = predicted_classes[i]

    key = f"{class_names[real]} → {class_names[pred]}"

    # on stocke seulement une erreur par type
    if key not in sample_errors:
        sample_errors[key] = i

    # si on a 6 cas → stop
    if len(sample_errors) == 6:
        break

print("\nExemples d'erreurs trouvées :")
for k, idx in sample_errors.items():
    print("-", k, "(index :", idx, ")")

# On affiche ensuite les images mal classées
for k, idx in sample_errors.items():
    img_path = test_itr.filepaths[idx]
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Mauvaise classification : {k}")
    plt.axis('off')

    # téléchargement des images mal classées
    filename = f"erreur_{k.replace(' ', '_').replace('→','_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# MATRICE D’IMAGES MAL CLASSÉES (3x3)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

class_names = ["baleine", "requin", "requinbaleine"]

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        idx_path = error_matrix[i][j]

        if idx_path is not None:
            img = plt.imread(idx_path)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            ax.imshow(img)
        else:
            ax.imshow(np.ones((100, 100, 3)))

        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(class_names[j])
        if j == 0:
            ax.set_ylabel(class_names[i])

plt.tight_layout()
fig.savefig("matrice_images_mal_classees.png", dpi=300, bbox_inches='tight')

plt.show()
plt.close(fig)
