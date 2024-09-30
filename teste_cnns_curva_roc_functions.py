import rasterio
from rasterio.features import rasterize
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, utils
from rasterio.features import shapes
import fiona
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def get_training_variables():

    raster_path = 'Imagem_landsat/bands_4_5_6.tif'

    shapefile = 'Recorte_Prodes/training_areas.shp' 
    polys = gpd.read_file(shapefile)

    with rasterio.open(raster_path) as src:
        
        bands = src.read([1, 2, 3])
        bands = np.moveaxis(bands, 0, -1)  

        if src.crs != polys.crs:
            print("Reprojetando shapefile para corresponder ao CRS do raster")
            polys = polys.to_crs(src.crs)


        unique_classes = polys['sub_class'].unique()
        class_mapping = {cls: i+1 for i, cls in enumerate(unique_classes)}
        print("Class_mapping:", class_mapping)
        y = rasterize(
            [(geom, class_mapping[cls]) for geom, cls in zip(polys.geometry, polys['sub_class'])],
            out_shape=(src.height, src.width),  
            transform=src.transform,
            fill=0,
            dtype='uint8'
        )

    X = bands[y > 0]
    y_flat = y[y > 0]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))
    print("Distribuição dos rótulos:", np.bincount(y_flat))

    return y, bands, src

def extract_patches(image, mask, patch_size=64, max_patches_per_class=100):
    patches = []
    labels = []
    class_count = {1: 0, 2: 0, 3: 0, 4: 0}
    height, width, _ = image.shape
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patch_mask = mask[i:i + patch_size, j:j + patch_size]

            if np.any(patch_mask > 0): 
                label = np.bincount(patch_mask.flat).argmax()

                if label == 0:
                        continue 

                if class_count[label] < max_patches_per_class:
                        patches.append(patch)
                        labels.append(label)
                        class_count[label] += 1
            
    for class_id in [1, 2, 3, 4]:
        if class_count[class_id] == 0:
            print(f"Warning: No patches extracted for class {class_id}")

    print("Classes dos patches:", np.unique(labels))
    print("Distribuição dos rótulos no treino:", np.bincount(labels))
    return np.array(patches), np.array(labels)

def cnn_model_definition(patches, labels_onehot):

    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))  # Usar Input() como a primeira camada
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    patches_train, validation_patches, labels_train, validation_labels_onehot = train_test_split(
        patches, labels_onehot, test_size=0.2, random_state=42
    )
    class_counts = np.bincount(np.argmax(labels_train, axis=-1))
    class_weights = {i: len(labels_train) / (class_counts[i] if class_counts[i] > 0 else 1) for i in range(len(class_counts))}

    
    # Treinamento
    # model.fit(patches, labels_onehot, epochs=10, batch_size=32)
    model.fit(patches_train, labels_train, epochs=10, batch_size=32, class_weight=class_weights)

    val_loss, val_accuracy = model.evaluate(validation_patches, validation_labels_onehot)
    print(f"Validação - Loss: {val_loss}, Accuracy: {val_accuracy}")

    y_pred_proba = model.predict(validation_patches)

    predicted_classes = np.argmax(y_pred_proba, axis=1)

    # Imprimir a distribuição das previsões
    unique, counts = np.unique(predicted_classes, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("Distribuição das previsões:", distribution)

    y_test_bin = label_binarize(np.argmax(validation_labels_onehot, axis=-1), classes=[0, 1, 2, 3])
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(4):  # 4 classes
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotar a curva ROC
    plt.figure()
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label='Classe {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC para cada classe')
    plt.legend(loc="lower right")
    plt.show()


    return model

def classify_image(model, image, patch_size=64):
    height, width, _ = image.shape
    classified_image = np.zeros((height, width), dtype=np.uint8)

    patches = []
    positions = []

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]

            # Verificar se o patch tem o tamanho correto (caso esteja nas bordas)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # Ignorar bordas

            patches.append(patch)
            positions.append((i, j))

    patches = np.array(patches)
    
    patches = patches.reshape(-1, patch_size, patch_size, 3)

    predictions = model.predict(patches)
    print("Classes previstas:", np.unique(predictions))

    predicted_classes = np.argmax(predictions, axis=-1)

    for idx, (i, j) in enumerate(positions):
        classified_image[i:i + patch_size, j:j + patch_size] = predicted_classes[idx] + 1
        
    return classified_image


def export_classified_to_shapefile(classified_image, transform, crs, output_shapefile):
    # Converter a imagem classificada para polígonos
    mask = classified_image > 0  # Criar máscara para áreas classificadas
    print("Máscara criada:", np.unique(classified_image[mask]))
    shapes_gen = shapes(classified_image, mask=mask, transform=transform)

    # Criar shapefile
    schema = {'geometry': 'Polygon', 'properties': {'class': 'int'}}
    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema=schema, crs=crs) as output:
        for geom, value in shapes_gen:
            output.write({'geometry': geom, 'properties': {'class': int(value)}})

def main():
    y, bands, src = get_training_variables()

    patches, labels = extract_patches(bands, y)

    print("Patches shape:", patches.shape)
    print("Labels shape:", labels.shape)
    print("Valores únicos das labels:", np.unique(labels))
    labels_onehot = utils.to_categorical(labels - 1, num_classes=4)
    
    print("Distribuição dos rótulos no treinamento:", np.bincount(labels))
    model = cnn_model_definition(patches, labels_onehot)

    start_time = time.time()
    classified_image = classify_image(model, bands)

    print("Valores únicos da imagem classificada:", np.unique(classified_image))

    end_time = time.time()
    classification_time = end_time - start_time

    print("Tempo de classificação:", classification_time)

    if classified_image.max() > 0:
        classified_image_scaled = (classified_image - classified_image.min()) * (255 / (classified_image.max() - classified_image.min()))
        classified_image_scaled = classified_image_scaled.astype(np.uint8)
    else:
        print("A imagem classificada não possui valores válidos para normalização.")

    plt.imshow(classified_image_scaled, cmap='nipy_spectral')
    plt.title("Classificação da Imagem")
    plt.show()
    
    export_classified_to_shapefile(classified_image, src.transform, src.crs, 'classified_output_cnnxxsc_class.shp')


main()