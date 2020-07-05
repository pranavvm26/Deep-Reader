import h5py
import os
from data_prep.prepare_dataset_words import Dataset



if __name__ == "__main__":
    INPUT_IMAGE_SIZE = (128, 64, 1)

    datagenerator = Dataset(source="/Users/pranavvm26/Git-Projects/HTR-DeepLearning/raw/iam",
                            destination="/Users/pranavvm26/PycharmProjects/HTRDeepLearning/hdf5_records",
                            name="iam")
    datagenerator.read_partitions()
    datagenerator.preprocess_partitions(input_size=INPUT_IMAGE_SIZE)

    for i in datagenerator.partitions:
        with h5py.File(os.path.join(datagenerator.destination, f"{datagenerator.name}_tvt.hdf5"), "a") as hf:
            hf.create_dataset(f"{i}/dt", data=datagenerator.dataset[i]['dt'],
                              compression="gzip", compression_opts=9)
            hf.create_dataset(f"{i}/gt", data=datagenerator.dataset[i]['gt'],
                              compression="gzip", compression_opts=9)
            print(f"[OK] {i} partition.")