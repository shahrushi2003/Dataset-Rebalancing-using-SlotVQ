# slot_run_name=
# orig_classifier_run_name=$orig_classifier_run_name
# rebal_classifier_run_name=
# dataset_name="mnist"
# logging="off"


# run script
cd training
python3 train_vsumm.py \
	--slot_run_name=$slot_run_name \
    --orig_classifier_run_name=$orig_classifier_run_name \
    --rebal_classifier_run_name=$rebal_classifier_run_name \
    --dataset_name=$dataset_name \
    --logging=$logging \
