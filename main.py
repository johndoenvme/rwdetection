from Misc import Utils
import os
CUR_DIR = os.getcwd()

DATA_PATH = os.path.join(CUR_DIR, 'CLEAR_Dataset')
DEMO_DATA_PATH = os.path.join(CUR_DIR, 'CLEAR_Dataset_Demo')
MODEL = 'CLT'  # 'RF', 'DeftPunk', 'PLT', 'CLT'
DEMO_MODE = True


def execute_experiment():
    #  Read and adapt data
    data_list = Utils.read_data(DEMO_DATA_PATH, MODEL)  # Change to DATA_PATH for full execution

    #  Create chunks
    data_with_chunks_list = Utils.run_chunker(data_list, MODEL)

    #  Preprocess according to the model
    if DEMO_MODE:
        data_with_chunks_list = Utils.extract_train_and_test_for_demo_mode(data_with_chunks_list)
    data_with_features_list = Utils.run_preprocessing(data_with_chunks_list, MODEL)

    #  Train and Infer
    results = Utils.train_and_infer(data_with_features_list, MODEL, DEMO_MODE)

    print(results)


if __name__ == '__main__':
    execute_experiment()
