import pickle
import numpy as np
from pathlib import Path

current_file_path = Path(__file__).resolve()
PATH_TO_NON_LW_DATA = str(current_file_path.parent / 'dataset' / 'data' / 'dataset.pkl')
PATH_TO_LW_DATA = str(current_file_path.parent / 'dataset' / 'data' / 'lw_dataset.pkl')
PATH_TO_LW_EXTRA_NPY_REPO_0 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_first_14k_articles.npy')
PATH_TO_LW_EXTRA_NPY_REPO_1 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_0-10000.npy')
PATH_TO_LW_EXTRA_NPY_REPO_2 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_10000-20000.npy')
PATH_TO_LW_EXTRA_NPY_REPO_3 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_20000-30000.npy')
PATH_TO_LW_EXTRA_NPY_REPO_4 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_30000-40000.npy')
PATH_TO_LW_EXTRA_NPY_REPO_5 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_40000-50000.npy')
PATH_TO_LW_EXTRA_NPY_REPO_6 = str(current_file_path.parent / 'dataset' / 'data' / 'embeddings_lw' / 'embeddings_50000-59554.npy')
npy_dataset = [PATH_TO_LW_EXTRA_NPY_REPO_0, PATH_TO_LW_EXTRA_NPY_REPO_1, PATH_TO_LW_EXTRA_NPY_REPO_2, PATH_TO_LW_EXTRA_NPY_REPO_3, PATH_TO_LW_EXTRA_NPY_REPO_4, PATH_TO_LW_EXTRA_NPY_REPO_5, PATH_TO_LW_EXTRA_NPY_REPO_6]

# with open(PATH_TO_LW_DATA, 'rb') as f:
#     dataset = pickle.load(f)
#     embeddings = np.load(PATH_TO_LW_EXTRA_NPY_REPO_0)
#     for i in range(1, len(npy_dataset)):
#         embeddings = np.concatenate((embeddings, np.load(npy_dataset[i])))
#     print(embeddings.shape)
#     print(len(dataset["embedding_strings"]))
#     dataset["embeddings"] = embeddings.astype(np.float32)
#
#
#
#     # dataset.save_data(PATH_TO_LW_DATA)
#     with open(PATH_TO_LW_DATA, 'wb') as f:
#         pickle.dump(dataset, f)
#
#
#
#

with open(PATH_TO_NON_LW_DATA, 'rb') as f_nlw, open(PATH_TO_LW_DATA, 'rb') as f_lw:
    dataset_nlw = pickle.load(f_nlw)  # to become dataset_all
    dataset_lw = pickle.load(f_lw)
    print(dataset_nlw["embeddings"].shape)
    print(dataset_lw["embeddings"].shape)
    
    dataset_nlw["metadata"] = dataset_nlw["metadata"] + dataset_lw["metadata"]
    dataset_nlw["embedding_strings"] = dataset_nlw["embedding_strings"] + dataset_lw["embedding_strings"]
    dataset_nlw["embeddings_metadata_index"] = dataset_nlw["embeddings_metadata_index"] + dataset_lw["embeddings_metadata_index"]
    dataset_nlw["embeddings"] = np.concatenate((dataset_nlw["embeddings"], dataset_lw["embeddings"]))

    dataset_nlw["total_articles_count"] = dataset_nlw["total_articles_count"] + dataset_lw["total_articles_count"]
    dataset_nlw["total_char_count"] = dataset_nlw["total_char_count"] + dataset_lw["total_char_count"]
    dataset_nlw["total_word_count"] = dataset_nlw["total_word_count"] + dataset_lw["total_word_count"]
    dataset_nlw["total_sentence_count"] = dataset_nlw["total_sentence_count"] + dataset_lw["total_sentence_count"]
    dataset_nlw["total_block_count"] = dataset_nlw["total_block_count"] + dataset_lw["total_block_count"]
    
    print(f"Embeddings shape: {dataset_nlw['embeddings'].shape}")
    print(f"Embedding strings length: {len(dataset_nlw['embedding_strings'])}")
    print(f"Embeddings metadata index length: {len(dataset_nlw['embeddings_metadata_index'])}")
    print(f"Total articles count: {dataset_nlw['total_articles_count']}")
    print(f"Total char count: {dataset_nlw['total_char_count']}")
    print(f"Total word count: {dataset_nlw['total_word_count']}")
    print(f"Total sentence count: {dataset_nlw['total_sentence_count']}")
    print(f"Total block count: {dataset_nlw['total_block_count']}")
    
    # dataset_nlw.save_data(PATH_TO_NON_LW_DATA)

    with open(PATH_TO_NON_LW_DATA, 'wb') as f:
        pickle.dump(dataset_nlw, f)

