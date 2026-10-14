# =========================================================
# cfd_data_tools.py
# =========================================================

from pathlib import Path
import json
import platform

import numpy as np
import h5py
import tensorflow as tf


# =========================================================
# Class 1: CFDTimeSeriesLoader
# =========================================================

class CFDTimeSeriesLoader:
    """
    Utility class for loading CFD time-series data and preparing it
    for machine learning.

    Main use cases:
        1. Read raw HDF5/STP CFD files.
        2. Extract one time signal at a selected y index.
        3. Save/load processed time series.
        4. Create forecasting windows.
        5. Create VAE windows.
        6. Split into train/validation/test.
        7. Normalize using training statistics only.
        8. Create tf.data.Dataset objects.
    """

    def __init__(
        self,
        data_dir=None,
        processed_dir="processed_data",
        extension=".stp",
        dataset_name="sta/utmp",
        time_name="header/timev",
        y_name="header/y",
        y_index=50,
        variable="utmp"
    ):
        self.system_name = platform.system()
        self.in_colab = self._detect_colab()

        if data_dir is None:
            # Default: local data folder next to the notebook/script
            self.data_dir = Path.cwd() / "data"
        else:
            self.data_dir = Path(data_dir)

        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)

        self.extension = extension
        self.dataset_name = dataset_name
        self.time_name = time_name
        self.y_name = y_name
        self.y_index = y_index
        self.variable = variable

        # These will be filled after loading or extracting data
        self.source_files = []
        self.all_timev = None
        self.all_signal = None
        self.all_file_indices = None
        self.sample_index = None

        self.file_names = None
        self.file_lengths = None
        self.file_start_indices = None

        self.y = None
        self.selected_y_value = None

        # Normalization parameters
        self.signal_mean = None
        self.signal_std = None

    # -----------------------------------------------------
    # Environment helpers
    # -----------------------------------------------------

    @staticmethod
    def _detect_colab():
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return False

    def print_environment(self):
        print("Operating system:", self.system_name)
        print("Running in Google Colab:", self.in_colab)
        print("Current working directory:", Path.cwd())
        print("Selected data directory:", self.data_dir)

        if not self.data_dir.exists():
            print("\nData directory not found.")

            if self.in_colab:
                print("\nIf your data are uploaded directly to Colab, use:")
                print('data_dir = Path("/content/data")')

                print("\nIf your data are in Google Drive, run:")
                print("from google.colab import drive")
                print('drive.mount("/content/drive")')
                print('data_dir = Path("/content/drive/<Drive folder>/data")')

        else:
            print("\nData directory found.")

    # -----------------------------------------------------
    # File discovery
    # -----------------------------------------------------

    def list_source_files(self):
        """
        List source files with the selected extension.
        """

        self.source_files = sorted(list(self.data_dir.glob(f"*{self.extension}")))

        print("Data directory:", self.data_dir)
        print(f"Number of {self.extension} files found:", len(self.source_files))

        for file in self.source_files[:10]:
            print(" -", file.name)

        if len(self.source_files) > 10:
            print(" ...")

        return self.source_files

    # -----------------------------------------------------
    # Extract signal from raw HDF5/STP files
    # -----------------------------------------------------

    def extract_signal_from_raw_files(self, max_files=None):
        """
        Read several HDF5/STP files and extract one time signal:

            signal = dataset[:, y_index]

        The final result is a concatenated time series.
        """

        if len(self.source_files) == 0:
            self.list_source_files()

        files_to_read = self.source_files

        if max_files is not None:
            files_to_read = files_to_read[:max_files]

        if len(files_to_read) == 0:
            raise FileNotFoundError(
                f"No files with extension {self.extension} found in {self.data_dir}"
            )

        time_list = []
        signal_list = []
        file_index_list = []

        self.file_names = []
        self.file_lengths = []
        self.file_start_indices = []

        current_start = 0

        for file_id, file_path in enumerate(files_to_read):

            with h5py.File(file_path, "r") as f:

                timev = f[self.time_name][:]
                y = f[self.y_name][:]
                signal = f[self.dataset_name][:, self.y_index]

            if self.y is None:
                self.y = y
                self.selected_y_value = float(y[self.y_index])

            time_list.append(timev)
            signal_list.append(signal)
            file_index_list.append(
                np.full(len(timev), file_id, dtype=np.int32)
            )

            self.file_names.append(file_path.name)
            self.file_lengths.append(len(timev))
            self.file_start_indices.append(current_start)

            current_start += len(timev)

            print(f"Read file {file_id}: {file_path.name}, samples: {len(timev)}")

        self.all_timev = np.concatenate(time_list).astype(np.float64)
        self.all_signal = np.concatenate(signal_list).astype(np.float32)
        self.all_file_indices = np.concatenate(file_index_list).astype(np.int32)
        self.sample_index = np.arange(len(self.all_signal), dtype=np.int64)

        self.file_names = np.asarray(self.file_names, dtype=object)
        self.file_lengths = np.asarray(self.file_lengths, dtype=np.int64)
        self.file_start_indices = np.asarray(self.file_start_indices, dtype=np.int64)

        print("\nFinal arrays:")
        print("all_timev:", self.all_timev.shape)
        print("all_signal:", self.all_signal.shape)
        print("all_file_indices:", self.all_file_indices.shape)
        print("sample_index:", self.sample_index.shape)
        print("selected_y_value:", self.selected_y_value)

        return self.all_timev, self.all_signal

    # -----------------------------------------------------
    # Save/load processed HDF5
    # -----------------------------------------------------

    def default_processed_h5_path(self):
        return self.processed_dir / f"{self.variable}_y{self.y_index}_raw_timeseries.h5"

    def save_processed_h5(self, output_file=None):
        """
        Save the processed raw time series as HDF5.
        """

        if self.all_signal is None:
            raise ValueError("No signal loaded. Run extract_signal_from_raw_files first.")

        if output_file is None:
            output_file = self.default_processed_h5_path()
        else:
            output_file = Path(output_file)

        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_file, "w") as f:

            raw_group = f.create_group("raw")

            raw_group.create_dataset(
                "timev",
                data=self.all_timev.astype(np.float64),
                compression="gzip",
                compression_opts=4
            )

            raw_group.create_dataset(
                "signal",
                data=self.all_signal.astype(np.float32),
                compression="gzip",
                compression_opts=4
            )

            raw_group.create_dataset(
                "file_indices",
                data=self.all_file_indices.astype(np.int32),
                compression="gzip",
                compression_opts=4
            )

            raw_group.create_dataset(
                "sample_index",
                data=self.sample_index.astype(np.int64),
                compression="gzip",
                compression_opts=4
            )

            files_group = f.create_group("files")

            files_group.create_dataset(
                "names",
                data=self.file_names,
                dtype=string_dtype
            )

            files_group.create_dataset(
                "lengths",
                data=self.file_lengths.astype(np.int64)
            )

            files_group.create_dataset(
                "start_indices",
                data=self.file_start_indices.astype(np.int64)
            )

            if self.y is not None:
                f.create_dataset(
                    "y",
                    data=self.y.astype(np.float64),
                    compression="gzip",
                    compression_opts=4
                )

            f.attrs["description"] = "Processed raw CFD time series for machine learning"
            f.attrs["variable"] = self.variable
            f.attrs["original_dataset"] = self.dataset_name
            f.attrs["time_dataset"] = self.time_name
            f.attrs["y_dataset"] = self.y_name
            f.attrs["y_index"] = int(self.y_index)
            f.attrs["selected_y_value"] = float(self.selected_y_value)
            f.attrs["n_total_samples"] = int(len(self.all_signal))
            f.attrs["n_files"] = int(len(self.file_names))

        print("Processed HDF5 saved to:")
        print(output_file)

        return output_file

    def load_processed_h5(self, input_file=None):
        """
        Load a processed HDF5 time-series file.
        """

        if input_file is None:
            input_file = self.default_processed_h5_path()
        else:
            input_file = Path(input_file)

        with h5py.File(input_file, "r") as f:

            self.all_timev = f["raw/timev"][:]
            self.all_signal = f["raw/signal"][:]
            self.all_file_indices = f["raw/file_indices"][:]
            self.sample_index = f["raw/sample_index"][:]

            self.file_names = f["files/names"][:]
            self.file_lengths = f["files/lengths"][:]
            self.file_start_indices = f["files/start_indices"][:]

            if "y" in f:
                self.y = f["y"][:]

            self.variable = f.attrs.get("variable", self.variable)
            self.dataset_name = f.attrs.get("original_dataset", self.dataset_name)
            self.time_name = f.attrs.get("time_dataset", self.time_name)
            self.y_name = f.attrs.get("y_dataset", self.y_name)
            self.y_index = int(f.attrs.get("y_index", self.y_index))
            self.selected_y_value = float(
                f.attrs.get("selected_y_value", np.nan)
            )

        print("Processed HDF5 loaded from:")
        print(input_file)

        print("\nLoaded arrays:")
        print("all_signal:", self.all_signal.shape)
        print("all_timev:", self.all_timev.shape)
        print("all_file_indices:", self.all_file_indices.shape)

        return self.all_timev, self.all_signal

    # -----------------------------------------------------
    # Save/load NPZ
    # -----------------------------------------------------

    def save_processed_npz(self, output_file=None):
        """
        Save processed raw time series as compressed NPZ.
        """

        if self.all_signal is None:
            raise ValueError("No signal loaded.")

        if output_file is None:
            output_file = self.processed_dir / f"{self.variable}_y{self.y_index}_raw_timeseries.npz"
        else:
            output_file = Path(output_file)

        np.savez_compressed(
            output_file,
            all_timev=self.all_timev,
            all_signal=self.all_signal,
            all_file_indices=self.all_file_indices,
            sample_index=self.sample_index,
            file_lengths=self.file_lengths,
            file_start_indices=self.file_start_indices,
            y_index=np.array(self.y_index, dtype=np.int32),
            selected_y_value=np.array(self.selected_y_value, dtype=np.float64),
            variable=self.variable
        )

        print("Processed NPZ saved to:")
        print(output_file)

        return output_file

    def load_processed_npz(self, input_file):
        """
        Load processed raw time series from NPZ.
        """

        input_file = Path(input_file)

        data = np.load(input_file)

        self.all_timev = data["all_timev"]
        self.all_signal = data["all_signal"]
        self.all_file_indices = data["all_file_indices"]
        self.sample_index = data["sample_index"]

        self.file_lengths = data["file_lengths"]
        self.file_start_indices = data["file_start_indices"]

        self.y_index = int(data["y_index"])
        self.selected_y_value = float(data["selected_y_value"])
        self.variable = str(data["variable"])

        print("Processed NPZ loaded from:")
        print(input_file)

        return self.all_timev, self.all_signal

    # -----------------------------------------------------
    # Window creation
    # -----------------------------------------------------

    def create_forecasting_windows(
        self,
        input_window=300,
        output_window=10,
        stride=10,
        avoid_file_boundaries=True
    ):
        """
        Create windows for forecasting.

        X shape:
            (samples, input_window, 1)

        y shape:
            (samples, output_window)

        Example:
            300 past values -> 10 future values
        """

        if self.all_signal is None:
            raise ValueError("No signal loaded.")

        signal = np.asarray(self.all_signal, dtype=np.float32)

        X_list = []
        y_list = []

        n_time = len(signal)
        last_start = n_time - input_window - output_window

        for start in range(0, last_start, stride):

            input_end = start + input_window
            output_end = input_end + output_window

            if avoid_file_boundaries and self.all_file_indices is not None:
                same_file = (
                    self.all_file_indices[start] == self.all_file_indices[input_end - 1]
                    and self.all_file_indices[start] == self.all_file_indices[output_end - 1]
                )

                if not same_file:
                    continue

            X_window = signal[start:input_end]
            y_window = signal[input_end:output_end]

            X_list.append(X_window)
            y_list.append(y_window)

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)

        X = X[..., np.newaxis]

        print("Forecasting windows created:")
        print("X:", X.shape)
        print("y:", y.shape)

        return X, y

    def create_vae_windows(
        self,
        window_size=300,
        stride=10,
        avoid_file_boundaries=True
    ):
        """
        Create windows for VAE / beta-VAE training.

        X shape:
            (samples, window_size, 1)

        The VAE input and target are the same.
        """

        if self.all_signal is None:
            raise ValueError("No signal loaded.")

        signal = np.asarray(self.all_signal, dtype=np.float32)

        X_list = []

        n_time = len(signal)
        last_start = n_time - window_size

        for start in range(0, last_start, stride):

            end = start + window_size

            if avoid_file_boundaries and self.all_file_indices is not None:
                if self.all_file_indices[start] != self.all_file_indices[end - 1]:
                    continue

            X_list.append(signal[start:end])

        X = np.asarray(X_list, dtype=np.float32)
        X = X[..., np.newaxis]

        print("VAE windows created:")
        print("X:", X.shape)

        return X

    # -----------------------------------------------------
    # Split and normalization
    # -----------------------------------------------------

    @staticmethod
    def split_train_val_test(X, y=None, train_fraction=0.70, val_fraction=0.15):
        """
        Chronological split.
        """

        n_samples = len(X)

        n_train = int(train_fraction * n_samples)
        n_val = int(val_fraction * n_samples)

        X_train = X[:n_train]
        X_val = X[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]

        if y is None:
            return X_train, X_val, X_test

        y_train = y[:n_train]
        y_val = y[n_train:n_train + n_val]
        y_test = y[n_train + n_val:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize_using_training(self, X_train, X_val, X_test, y_train=None, y_val=None, y_test=None):
        """
        Normalize X and y using statistics from X_train only.
        """

        self.signal_mean = X_train.mean()
        self.signal_std = X_train.std()

        if self.signal_std == 0:
            self.signal_std = 1.0

        X_train_norm = (X_train - self.signal_mean) / self.signal_std
        X_val_norm = (X_val - self.signal_mean) / self.signal_std
        X_test_norm = (X_test - self.signal_mean) / self.signal_std

        print("Normalization statistics:")
        print("signal_mean:", self.signal_mean)
        print("signal_std:", self.signal_std)

        if y_train is None:
            return X_train_norm, X_val_norm, X_test_norm

        y_train_norm = (y_train - self.signal_mean) / self.signal_std
        y_val_norm = (y_val - self.signal_mean) / self.signal_std
        y_test_norm = (y_test - self.signal_mean) / self.signal_std

        return (
            X_train_norm,
            X_val_norm,
            X_test_norm,
            y_train_norm,
            y_val_norm,
            y_test_norm
        )

    @staticmethod
    def make_tf_datasets(
        X_train,
        X_val,
        X_test,
        y_train=None,
        y_val=None,
        y_test=None,
        batch_size=64,
        shuffle=True,
        seed=0
    ):
        """
        Create tf.data.Dataset objects.
        """

        AUTOTUNE = tf.data.AUTOTUNE

        if y_train is None:
            train_ds = tf.data.Dataset.from_tensor_slices(X_train)
            val_ds = tf.data.Dataset.from_tensor_slices(X_val)
            test_ds = tf.data.Dataset.from_tensor_slices(X_test)
        else:
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        if shuffle:
            train_ds = train_ds.shuffle(
                buffer_size=len(X_train),
                seed=seed
            )

        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

        return train_ds, val_ds, test_ds

    # -----------------------------------------------------
    # High-level workflows
    # -----------------------------------------------------

    def prepare_forecasting_data(
        self,
        input_window=300,
        output_window=10,
        stride=10,
        batch_size=64,
        train_fraction=0.70,
        val_fraction=0.15,
        shuffle=True
    ):
        """
        Complete forecasting workflow.

        Returns a dictionary with:
            X_train_norm, y_train_norm, ...
            train_ds, val_ds, test_ds
            normalization statistics
        """

        X, y = self.create_forecasting_windows(
            input_window=input_window,
            output_window=output_window,
            stride=stride
        )

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = self.split_train_val_test(
            X,
            y,
            train_fraction=train_fraction,
            val_fraction=val_fraction
        )

        (
            X_train_norm,
            X_val_norm,
            X_test_norm,
            y_train_norm,
            y_val_norm,
            y_test_norm
        ) = self.normalize_using_training(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        train_ds, val_ds, test_ds = self.make_tf_datasets(
            X_train_norm,
            X_val_norm,
            X_test_norm,
            y_train_norm,
            y_val_norm,
            y_test_norm,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return {
            "X_train": X_train_norm,
            "X_val": X_val_norm,
            "X_test": X_test_norm,
            "y_train": y_train_norm,
            "y_val": y_val_norm,
            "y_test": y_test_norm,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "signal_mean": self.signal_mean,
            "signal_std": self.signal_std,
            "input_window": input_window,
            "output_window": output_window,
            "stride": stride
        }

    def prepare_vae_data(
        self,
        window_size=300,
        stride=10,
        batch_size=64,
        train_fraction=0.70,
        val_fraction=0.15,
        shuffle=True
    ):
        """
        Complete VAE / beta-VAE workflow.
        """

        X = self.create_vae_windows(
            window_size=window_size,
            stride=stride
        )

        X_train, X_val, X_test = self.split_train_val_test(
            X,
            y=None,
            train_fraction=train_fraction,
            val_fraction=val_fraction
        )

        X_train_norm, X_val_norm, X_test_norm = self.normalize_using_training(
            X_train,
            X_val,
            X_test
        )

        train_ds, val_ds, test_ds = self.make_tf_datasets(
            X_train_norm,
            X_val_norm,
            X_test_norm,
            y_train=None,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return {
            "X_train": X_train_norm,
            "X_val": X_val_norm,
            "X_test": X_test_norm,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "signal_mean": self.signal_mean,
            "signal_std": self.signal_std,
            "window_size": window_size,
            "stride": stride
        }


# =========================================================
# Class 2: H5ToTFRecordConverter
# =========================================================

class H5ToTFRecordConverter:
    """
    Convert processed HDF5 time-series data into TFRecords.

    It supports two strategies:

    1. Raw TFRecords:
        One record per original CFD file segment.

    2. Window TFRecords:
        One record per machine learning sample:
            X = past window
            y = future window
    """

    def __init__(
        self,
        h5_file,
        output_dir=None,
        compression_type=""
    ):
        self.h5_file = Path(h5_file)

        if output_dir is None:
            self.output_dir = self.h5_file.parent / (self.h5_file.stem + "_tfrecords")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)

        self.compression_type = compression_type

        if compression_type == "GZIP":
            self.file_suffix = ".tfrecord.gz"
            self.writer_options = tf.io.TFRecordOptions(
                compression_type="GZIP"
            )
        else:
            self.file_suffix = ".tfrecord"
            self.writer_options = None

        self.metadata = {}

    # -----------------------------------------------------
    # TFRecord helper functions
    # -----------------------------------------------------

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, str):
            value = value.encode("utf-8")

        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(value)])
        )

    @staticmethod
    def _serialize_tensor(array):
        tensor = tf.convert_to_tensor(array)
        serialized_tensor = tf.io.serialize_tensor(tensor)
        return serialized_tensor.numpy()

    # -----------------------------------------------------
    # Load processed HDF5
    # -----------------------------------------------------

    def load_h5_arrays(self):
        """
        Load the processed HDF5 arrays into memory.

        This is fine for the reduced processed signal.
        For terabyte-scale data, use streaming from the raw files instead.
        """

        with h5py.File(self.h5_file, "r") as f:

            all_timev = f["raw/timev"][:]
            all_signal = f["raw/signal"][:]
            all_file_indices = f["raw/file_indices"][:]
            sample_index = f["raw/sample_index"][:]

            file_names = f["files/names"][:]
            file_lengths = f["files/lengths"][:]
            file_start_indices = f["files/start_indices"][:]

            attrs = dict(f.attrs)

        return {
            "all_timev": all_timev,
            "all_signal": all_signal,
            "all_file_indices": all_file_indices,
            "sample_index": sample_index,
            "file_names": file_names,
            "file_lengths": file_lengths,
            "file_start_indices": file_start_indices,
            "attrs": attrs
        }

    # -----------------------------------------------------
    # Raw TFRecords
    # -----------------------------------------------------

    def _create_raw_example(
        self,
        file_id,
        file_name,
        start_index,
        length,
        timev_segment,
        signal_segment,
        file_indices_segment,
        sample_index_segment
    ):
        feature = {
            "file_id": self._int64_feature(file_id),
            "file_name": self._bytes_feature(file_name),
            "start_index": self._int64_feature(start_index),
            "length": self._int64_feature(length),
            "timev": self._bytes_feature(self._serialize_tensor(timev_segment)),
            "signal": self._bytes_feature(self._serialize_tensor(signal_segment)),
            "file_indices": self._bytes_feature(self._serialize_tensor(file_indices_segment)),
            "sample_index": self._bytes_feature(self._serialize_tensor(sample_index_segment)),
        }

        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        )

    def write_raw_tfrecords(self):
        """
        Write one TFRecord file per original CFD file segment.
        """

        data = self.load_h5_arrays()

        all_timev = data["all_timev"]
        all_signal = data["all_signal"]
        all_file_indices = data["all_file_indices"]
        sample_index = data["sample_index"]
        file_names = data["file_names"]
        file_lengths = data["file_lengths"]
        file_start_indices = data["file_start_indices"]
        attrs = data["attrs"]

        shards = []

        for file_id in range(len(file_names)):

            start_index = int(file_start_indices[file_id])
            length = int(file_lengths[file_id])
            end_index = start_index + length

            file_name = file_names[file_id]

            if isinstance(file_name, bytes):
                file_name = file_name.decode("utf-8")

            tfrecord_file = self.output_dir / f"raw_{file_id:05d}{self.file_suffix}"

            example = self._create_raw_example(
                file_id=file_id,
                file_name=file_name,
                start_index=start_index,
                length=length,
                timev_segment=all_timev[start_index:end_index],
                signal_segment=all_signal[start_index:end_index],
                file_indices_segment=all_file_indices[start_index:end_index],
                sample_index_segment=sample_index[start_index:end_index]
            )

            with tf.io.TFRecordWriter(
                str(tfrecord_file),
                options=self.writer_options
            ) as writer:
                writer.write(example.SerializeToString())

            shards.append(
                {
                    "file_id": int(file_id),
                    "tfrecord_file": tfrecord_file.name,
                    "original_file_name": file_name,
                    "start_index": int(start_index),
                    "length": int(length)
                }
            )

            print("Saved:", tfrecord_file.name)

        self.metadata = {
            "description": "Raw time-series TFRecords",
            "source_h5_file": str(self.h5_file),
            "strategy": "one record per original CFD file segment",
            "compression_type": self.compression_type,
            "attrs": {k: str(v) for k, v in attrs.items()},
            "shards": shards
        }

        metadata_file = self.output_dir / "raw_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print("\nRaw TFRecord metadata saved to:")
        print(metadata_file)

        return self.output_dir

    # -----------------------------------------------------
    # Window TFRecords
    # -----------------------------------------------------

    def _create_window_example(
        self,
        sample_id,
        start_index,
        file_id,
        X_window,
        y_window
    ):
        feature = {
            "sample_id": self._int64_feature(sample_id),
            "start_index": self._int64_feature(start_index),
            "file_id": self._int64_feature(file_id),
            "X": self._bytes_feature(self._serialize_tensor(X_window.astype(np.float32))),
            "y": self._bytes_feature(self._serialize_tensor(y_window.astype(np.float32))),
        }

        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        )

    def write_window_tfrecords(
        self,
        input_window=300,
        output_window=10,
        stride=10,
        records_per_shard=512,
        avoid_file_boundaries=True
    ):
        """
        Write TFRecords directly as training samples.

        Each record contains:
            X = signal[start:start+input_window]
            y = signal[start+input_window:start+input_window+output_window]
        """

        data = self.load_h5_arrays()

        signal = data["all_signal"].astype(np.float32)
        file_indices = data["all_file_indices"].astype(np.int32)

        n_time = len(signal)
        last_start = n_time - input_window - output_window

        shard_id = 0
        records_in_current_shard = 0
        sample_id = 0
        writer = None

        shards = []

        def open_new_writer(shard_id):
            tfrecord_file = self.output_dir / f"windows_{shard_id:05d}{self.file_suffix}"

            writer = tf.io.TFRecordWriter(
                str(tfrecord_file),
                options=self.writer_options
            )

            shards.append(
                {
                    "shard_id": int(shard_id),
                    "tfrecord_file": tfrecord_file.name
                }
            )

            return writer, tfrecord_file

        writer, current_file = open_new_writer(shard_id)

        for start in range(0, last_start, stride):

            input_end = start + input_window
            output_end = input_end + output_window

            if avoid_file_boundaries:
                same_file = (
                    file_indices[start] == file_indices[input_end - 1]
                    and file_indices[start] == file_indices[output_end - 1]
                )

                if not same_file:
                    continue

            file_id = int(file_indices[start])

            X_window = signal[start:input_end]
            y_window = signal[input_end:output_end]

            example = self._create_window_example(
                sample_id=sample_id,
                start_index=start,
                file_id=file_id,
                X_window=X_window,
                y_window=y_window
            )

            writer.write(example.SerializeToString())

            sample_id += 1
            records_in_current_shard += 1

            if records_in_current_shard >= records_per_shard:
                writer.close()
                print("Saved:", current_file.name)

                shard_id += 1
                records_in_current_shard = 0
                writer, current_file = open_new_writer(shard_id)

        if writer is not None:
            writer.close()
            print("Saved:", current_file.name)

        metadata = {
            "description": "Windowed TFRecords for forecasting",
            "source_h5_file": str(self.h5_file),
            "strategy": "one record per forecasting sample",
            "input_window": int(input_window),
            "output_window": int(output_window),
            "stride": int(stride),
            "records_per_shard": int(records_per_shard),
            "n_samples": int(sample_id),
            "compression_type": self.compression_type,
            "record_schema": {
                "sample_id": "int64",
                "start_index": "int64",
                "file_id": "int64",
                "X": "serialized tensor, float32, shape (input_window,)",
                "y": "serialized tensor, float32, shape (output_window,)"
            },
            "shards": shards
        }

        metadata_file = self.output_dir / "windows_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        print("\nWindow TFRecord metadata saved to:")
        print(metadata_file)
        print("Number of samples:", sample_id)

        return self.output_dir

    # -----------------------------------------------------
    # Read window TFRecords
    # -----------------------------------------------------

    @staticmethod
    def _parse_window_example(serialized_example, input_window, output_window):
        feature_description = {
            "sample_id": tf.io.FixedLenFeature([], tf.int64),
            "start_index": tf.io.FixedLenFeature([], tf.int64),
            "file_id": tf.io.FixedLenFeature([], tf.int64),
            "X": tf.io.FixedLenFeature([], tf.string),
            "y": tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(
            serialized_example,
            feature_description
        )

        X = tf.io.parse_tensor(example["X"], out_type=tf.float32)
        y = tf.io.parse_tensor(example["y"], out_type=tf.float32)

        X = tf.reshape(X, (input_window, 1))
        y = tf.reshape(y, (output_window,))

        return X, y

    def load_window_tfrecord_dataset(
        self,
        input_window,
        output_window,
        batch_size=64,
        shuffle_buffer=2048,
        shuffle=True
    ):
        """
        Load windowed TFRecords as a tf.data.Dataset.
        """

        tfrecord_files = sorted(
            list(self.output_dir.glob("windows_*.tfrecord")) +
            list(self.output_dir.glob("windows_*.tfrecord.gz"))
        )

        if len(tfrecord_files) == 0:
            raise FileNotFoundError("No window TFRecord files found.")

        dataset = tf.data.TFRecordDataset(
            [str(file) for file in tfrecord_files],
            compression_type=self.compression_type
        )

        dataset = dataset.map(
            lambda x: self._parse_window_example(
                x,
                input_window=input_window,
                output_window=output_window
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset