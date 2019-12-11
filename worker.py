import numpy as np
import math
import pandas as pd
import geopandas as gpd
import rasterio
import pyproj
import os
from time import gmtime, strftime, time
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.mask import mask
from rasterio.windows import Window
from shapely.geometry import mapping
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QMainWindow
from sklearn.utils import parallel_backend, register_parallel_backend
from joblib._parallel_backends import ThreadingBackend
from sklearn.preprocessing import MinMaxScaler


class Signals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)
    status = pyqtSignal(str, str)

    def __init__(self, thread):
        super(Signals, self).__init__()

        self.thread = thread

    @pyqtSlot()
    def cancel(self):
        self.thread.stop()


class Worker(QThread):

    register_parallel_backend("threading", ThreadingBackend, make_default=True)

    def __init__(
        self,
        rlayer,
        vlayer,
        outlayer,
        fields,
        classifier,
        model_params,
        split_params,
        tiles,
        accass,
        max_pix,
        tr,
    ):
        super(Worker, self).__init__()
        self.signals = Signals(self)
        try:
            self.starttime = time()
            self.rlayer = rlayer
            self.vlayer = vlayer
            self.outlayer = outlayer
            self.fields = fields
            self.classif = classifier
            self.params = model_params
            self.sp_params = split_params
            self.tiles = tiles
            self.acc = accass
            self.max_pix = max_pix
            self.tr = tr

        except Exception as e:
            import traceback

            self.signals.error.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()), traceback.format_exc()
            )
            self.signals.finished.emit(None)

    def run(self):
        self.signals.status.emit(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Start...")
        try:

            self.train()
            result = self.classify()
            self.signals.finished.emit(result)

        except Exception as e:
            import traceback

            self.signals.error.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()), traceback.format_exc()
            )
            self.signals.finished.emit(None)

    def train(self):
        def split(X, Y, test_size, stratify):
            return train_test_split(X, Y, test_size=test_size, stratify=stratify)

        shp = self.vlayer.dataProvider().dataSourceUri()
        train_shp = shp.split("|")[0]
        self.signals.status.emit(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Open vector file: " + train_shp
        )

        shapefile = gpd.read_file(train_shp)

        labelencoder = LabelEncoder()

        #### TODO: special case self.fields == 'ClassificationTool_encoding'

        shapefile["ClassificationTool_encoding"] = labelencoder.fit_transform(
            shapefile[self.fields]
        )
        self.shp_stats = (
            shapefile.groupby([self.fields, "ClassificationTool_encoding"])
            .size()
            .reset_index()
            .rename(columns={0: "Number of Polygons"})
        )
        geoms = shapefile.geometry.values

        self.signals.status.emit(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "Open raster file: " + self.rlayer.dataProvider().dataSourceUri(),
        )
        with rasterio.open(self.rlayer.dataProvider().dataSourceUri()) as src:
            img_bands = src.count
            crs = src.crs

        p1 = pyproj.Proj(crs)
        p2 = pyproj.Proj(shapefile.crs)
        if p1.srs != p2.srs:
            raise RuntimeError("Error: data sets have different projections")

        X = np.array([]).reshape(0, img_bands)
        y = np.array([], dtype=np.int8)

        self.signals.status.emit(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Extract raster values ..."
        )
        with rasterio.open(self.rlayer.dataProvider().dataSourceUri()) as src:
            meta = src.meta
            for index, geom in enumerate(geoms):
                feature = [mapping(geom)]

                out_image, out_transform = mask(src, feature, crop=True)

                out_image_trimmed = out_image[
                    :, ~np.all(out_image == meta["nodata"], axis=0)
                ]
                out_image_reshaped = np.transpose(out_image_trimmed)
                if self.max_pix > -1:
                    out_image_reshaped = out_image_reshaped[np.random.choice(out_image_reshaped.shape[0], self.max_pix ,replace=False)]
                y = np.append(
                    y,
                    [shapefile["ClassificationTool_encoding"][index]]
                    * out_image_reshaped.shape[0],
                )

                X = np.vstack((X, out_image_reshaped))

        self.signals.status.emit(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "Split data into training and test subset: "
            + str(100 - (self.sp_params["test_size"]))
            + "% test data "
            + str(self.sp_params["test_size"])
            + "% training data",
        )

        if self.sp_params["stratify"] == True:
            stratify = y
        else:
            stratify = None

        test_size = self.sp_params["test_size"] / 100

        X_train, X_test, y_train, y_test = split(X, y, test_size, stratify)

        if self.classif == "KNearestNeighbor":
            self.signals.status.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "Train model using " + self.classif,
            )
            from PyQt5 import QtTest

            QtTest.QTest.qWait(30000)
            scaler = MinMaxScaler(feature_range=(0, 1))
            rescaledX = scaler.fit_transform(X)
            classifier = KNeighborsClassifier(**self.params)
            classifier.fit(X_train, y_train)

        if self.classif == "RandomForest":
            self.signals.status.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "Train model using " + self.classif,
            )
            classifier = RandomForestClassifier(**self.params)
            classifier.fit(X, y)

        if self.classif == "SVC":
            self.signals.status.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                "Train model using " + self.classif,
            )

            classifier = SVC(**self.params)
            classifier.fit(X_train, y_train)

        self.classifier = classifier

        if self.acc == True:
            y_pred = classifier.predict(X_test)
            cm = metrics.confusion_matrix(y_test, y_pred)
            stat_sort = self.shp_stats.sort_values("ClassificationTool_encoding")

            cmpd = pd.DataFrame(
                data=cm,
                index=stat_sort[self.fields].values,
                columns=stat_sort[self.fields].values,
            )
            cmpd_out = os.path.splitext(self.outlayer)[0] + "CM.csv"
            cmpd.to_csv(cmpd_out)

            clsf_report = pd.DataFrame(
                metrics.classification_report(
                    y_true=y_test, y_pred=y_pred, output_dict=True
                )
            ).transpose()
            clist = stat_sort[self.fields].to_list()
            clist.extend(["micro avg", "macro avg", "weighted avg"])
            clsf_report.index = clist
            clsf_out = os.path.splitext(self.outlayer)[0] + "class_report.csv"
            clsf_report.to_csv(clsf_out)

    def classify(self):
        def calculate_chunks(width, height, tiles):
            pixels = width * height
            max_pixels = pixels / tiles
            chunk_size = int(math.floor(math.sqrt(max_pixels)))
            ncols = int(math.ceil(width / chunk_size))
            nrows = int(math.ceil(height / chunk_size))
            chunk_windows = []

            for col in range(ncols):
                col_offset = col * chunk_size
                w = min(chunk_size, width - col_offset)
                for row in range(nrows):
                    row_offset = row * chunk_size
                    h = min(chunk_size, height - row_offset)
                    chunk_windows.append(
                        ((row, col), Window(col_offset, row_offset, w, h))
                    )
            return chunk_windows

        with rasterio.open(self.rlayer.dataProvider().dataSourceUri()) as src:
            width = src.width
            height = src.height
            bands = src.count
            meta = src.meta
            dtype = src.dtypes

            self.signals.status.emit(
                strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Predicting image values ... "
            )

            chunk_blocks = calculate_chunks(width, height, self.tiles)
            meta.update({"count": 1, "dtype": dtype[0]})

            with rasterio.open(self.outlayer, "w", **meta) as dst:
                counter = 1
                for idx, window in chunk_blocks:
                    self.signals.status.emit(
                        strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                        "Processing Block: "
                        + str(counter)
                        + " of "
                        + str(len(chunk_blocks))
                    )
                    img = src.read(window=window)
                    dtype = rasterio.dtypes.get_minimum_dtype(img)
                    reshaped_img = reshape_as_image(img)
                    rows, cols, bands_n = reshaped_img.shape

                    class_prediction = self.classifier.predict(
                        reshaped_img.reshape(-1, bands)
                    )
                    classification = np.zeros((rows, cols, 1)).astype(dtype)
                    classification[:, :, 0] = class_prediction.reshape(
                        reshaped_img[:, :, 1].shape
                    ).astype(dtype)
                    final = reshape_as_raster(classification)
                    dst.write(final, window=window)
                    counter += 1

        seconds_elapsed = time() - self.starttime
        self.signals.status.emit(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "Execution completed in "
            + str(np.around(seconds_elapsed, decimals=2))
            + " seconds",
        )
        return self.outlayer
