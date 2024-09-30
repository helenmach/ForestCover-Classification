import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_training_variables():
    raster_path = 'Imagem_landsat/bands_4_5_6.tif'
    polygon_shp_path = 'Recorte_Prodes/training_areas.shp' 
    polys = gpd.read_file(polygon_shp_path)

    with rasterio.open(raster_path) as src:
        
        bands = src.read([1, 2, 3])
        bands = np.moveaxis(bands, 0, -1)  

        if src.crs != polys.crs:
            print("Reprojetando shapefile para corresponder ao CRS do raster")
            polys = polys.to_crs(src.crs)

        X = bands.reshape(-1, bands.shape[-1])

        unique_classes = polys['sub_class'].unique()
        class_mapping = {cls: i+1 for i, cls in enumerate(unique_classes)}
        print("Mapeamento das classes:", class_mapping)
        
        y = rasterize(
            [(geom, class_mapping[cls]) for geom, cls in zip(polys.geometry, polys['sub_class'])],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype='uint8'
        ).flatten()

    X = X[y > 0]
    y = y[y > 0]

    print("Distribuição dos rótulos:", np.bincount(y))

    return X, y, bands, src, polys

def random_forest_trinig(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    print(f'Tempo de treinamento: {end_time - start_time:.2f} segundos')

    # y_pred = rf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Acurácia do modelo: {accuracy:.2f}')

    y_pred_proba = rf.predict_proba(X_test)
    print("Probabilidades das classes:", y_pred_proba)

    threshold = 0.5

    y_pred_thresholded = np.where(y_pred_proba[:, 1] >= threshold, 2, 1)
    print("Previsões com ponto de corte:", y_pred_thresholded)

    unique, counts = np.unique(y_pred_thresholded, return_counts=True)
    print("Distribuição das previsões:", dict(zip(unique, counts)))

    verify_matriz = matriz_de_confusao(y_test, y_pred_thresholded)

    accuracy = accuracy_score(y_test, y_pred_thresholded)
    print(f'Acurácia do modelo com ponto de corte {threshold}: {accuracy:.2f}')

    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred_bin = label_binarize(y_pred_thresholded, classes=np.unique(y))

    plot_roc(y_test_bin, y_pred_bin, 'Random Forest com Ponto de Corte')

    return rf
    
    
def random_forest_classify(rf, bands):
    start_time = time.time()
    raster_classified = rf.predict(bands.reshape(-1, bands.shape[-1]))

    raster_classified = raster_classified.reshape(bands.shape[:2])

    unique_classes, counts = np.unique(raster_classified, return_counts=True)

    for cls, count in zip(unique_classes, counts):
        print(f'Classe {cls}: {count} pixels')

    end_time = time.time()
    print(f'Tempo de classificação: {end_time - start_time:.2f} segundos')

    return raster_classified

def get_ground_truth(src):

    ground_truth = gpd.read_file('Recorte_Prodes/recorte_prodes_filtered_by_date.shp')

    if ground_truth.crs != src.crs:
        print("Reprojetando ground truth para corresponder ao CRS do raster classificado")
        ground_truth = ground_truth.to_crs(src.crs)

    ground_truth['class'] = np.where(
        ground_truth['sub_class'] == 'corte raso com vegetação', 1,
        np.where(ground_truth['sub_class'] == 'corte raso com solo exposto', 2, None)
    )

    ground_truth_filtered = ground_truth.dropna(subset=['class'])

    ground_truth_raster = rasterize(
        [(geom, int(cls)) for geom, cls in zip(ground_truth_filtered.geometry, ground_truth_filtered['class'])],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype='uint8'
    ).flatten()

    classes_ground_truth = ground_truth_filtered['class'].unique()
    print("Classes no ground truth:", classes_ground_truth)

    return ground_truth_raster

def validation_model_classification(raster_classified, ground_truth_raster):

    classified_raster = raster_classified.flatten()
    classified_raster_filtered = np.where(np.isin(classified_raster, [1, 2]), classified_raster, 0)

    mask = (ground_truth_raster > 0) & (classified_raster_filtered > 0)
    classified_raster_filtered = classified_raster_filtered[mask]
    ground_truth_raster_filtered = ground_truth_raster[mask]

    y_test_bin = label_binarize(ground_truth_raster_filtered, classes=[1, 2])
    y_pred_bin = label_binarize(classified_raster_filtered, classes=[1, 2])
    label = 'Validação do modelo'

    unique_pred_classes = np.unique(classified_raster_filtered)
    print("Classes previstas:", unique_pred_classes)

    plot_roc(y_test_bin, y_pred_bin, label)


    # if y_test_bin.shape[1] > 1:
    #     plot_roc(y_test_bin[:, 1], y_pred_bin[:, 1], 'Random Forest Final (Classes 1 e 2)')
    # else:
    #     plot_roc(y_test_bin, y_pred_bin, 'Random Forest Final (Classes 1 e 2)')



def matriz_de_confusao(y_test, y_pred_thresholded):
    cm = confusion_matrix(y_test, y_pred_thresholded)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2], yticklabels=[1, 2])
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Verdadeiras')
    plt.title('Matriz de Confusão')
    plt.show()


def plot_roc(y_true, y_pred, label):
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    auc_score = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{label} AUC: {auc_score:.2f}')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()


def main():
    X, y, bands, src, polys = get_training_variables()

    rf = random_forest_trinig(X, y)

    raster_classified = random_forest_classify(rf, bands)

    ground_truth_raster = get_ground_truth(src)

    validation_model_classification(raster_classified, ground_truth_raster)


    output_classified_path = 'random_forest_50_arvores_corte_05.tif'
    with rasterio.open(
        output_classified_path,
        'w',
        driver='GTiff',
        height=raster_classified.shape[0],
        width=raster_classified.shape[1],
        count=1,
        dtype=raster_classified.dtype,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(raster_classified, 1)

    # Converter o raster classificado para shapefile
    output_polygons_shp_path = 'random_forest_50_arvores_corte_05.shp'

    # Abrir a imagem classificada
    with rasterio.open(output_classified_path) as src:
        classified_array = src.read(1)
        transform = src.transform

    # Criar um GeoDataFrame contendo os polígonos dos valores classificados
    shapes_gen = shapes(classified_array, transform=transform)
    records = [{'geometry': shape, 'properties': {'class': value}} for shape, value in shapes_gen]

    # Salvar o GeoDataFrame como shapefile
    gdf = gpd.GeoDataFrame.from_features(records, crs=polys.crs)
    gdf.to_file(output_polygons_shp_path)


main()

