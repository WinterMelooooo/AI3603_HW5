RESULT_DIR="/home/yktang/AI3603_HW5/code/resultsCAT/LargeChannel_MedianDimMults_LargeBatchSize_LongEpoch/2024_12_22_16_21_47"
DATASET_DIR="/home/yktang/AI3603_HW5/data/faces"
pytorch-fid "${RESULT_DIR}/submission" "${DATASET_DIR}" > "${RESULT_DIR}/eval.txt"