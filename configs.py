data_args = {"base_dir" : "/content",
             "train_bias_conflicting_data_ratio" : 0.2,
             "test_bias_conflicting_data_ratio" : 0,
             "bias_type" : "background",
             "square_number" : 1,
             "use_random_masking" : False,
             "train_batch" : 64,
             "workers" : 2,
             "test_batch" : 64,
             "masking_batch_size" : 128,
             "test_data_types" : "background",
             "seed": 64}

static_model_configs = {
    "model_dir":'/content/',
    "num_workers":data_args["workers"],
    "num_classes":10,
    "resolution":(28, 28)
    }
hyp_model_configs = {
    "num_slots":3,
    "seed":42,
    "num_iterations":3,
    "batch_size":data_args["train_batch"],
    "hid_dim":32,
    "codebook_size":3,
    "beta":0.8,
    "use_kmeans":False,
    "z_norm":None,
    "cb_norm":None,
    "affine_lr":0.0,
    "sync_nu":0.0,
    "replace_freq":0,
    "learning_rate":3e-4,
    "num_epochs":20,
    "alpha":10,
    "temperature": 1
    }

model_args = static_model_configs | hyp_model_configs


static_classifier_configs = {
    "model_dir":'/content/',
    "seed":42,
    "num_workers":data_args["workers"],
    "num_classes":10,
    "resolution":model_args["resolution"]
    }
hyp_classifier_configs = {
    "model_type":"Conv",
    "batch_size":64,
    "learning_rate":1e-4,
    "num_epochs":10,
    }

classifier_args = static_classifier_configs | hyp_classifier_configs
