config = {
    "model": {
        "hooks": {
            "preinit_hooks": [
            ],
            "init_hooks": [
            ],
            "preprocess_hooks": [
            ],
            "postprocess_hooks": [
            ]
        }
    },
    "backbone": {
        "hooks": {
            "preinit_hooks": [
            ],
            "init_hooks": [
            ],
            "preprocess_hooks": [
            ],
            "postprocess_hooks": [
            ]
        },
        "type": "resnet",
        "cls": "ResNet50",
        "args": {
            "out_feats": {"index": 0, "value": [True, True, True, True, True]}
        },
        "kwargs": {
            
        }
    },
    "detector": {
        "hooks": {
            "preinit_hooks": [
            ],
            "init_hooks": [
            ],
            "preprocess_hooks": [
            ],
            "postprocess_hooks": [
            ]
        },
        "type": "retinanet",
        "cls": "RetinaNet",
        "args": {
            "num_classes": {"index": 1, "value": 80}
        },
        "kwargs": {
        }
    },
    "semantic_variables": [
        {
            "hooks": {
                "preinit_hooks": [
                ],
                "init_hooks": [
                ],
                "preprocess_hooks": [
                ],
                "postprocess_hooks": [
                ]
            },
            "category": "scene_recognition",
            "type": "densenet161",
            "cls": "DenseNet161",
            "args": {
                "num_classes": {"index": 2, "value": 365}
            },
            "kwargs": {
                "load_from": "densenet161_places365-62bbf0d4.pth"
            }
        },
        {
            "hooks": {
                "preinit_hooks": [
                ],
                "init_hooks": [
                ],
                "preprocess_hooks": [
                ],
                "postprocess_hooks": [
                ]
            },
            "category": "monocular_depth_estimation",
            "type": "glpdepth",
            "cls": "GLPDepth",
            "args": {

            },
            "kwargs": {
                
            }
        }          
    ],
    "output_head": {
        "hooks": {
            "preinit_hooks": [

            ],
            "init_hooks": [
                
            ],
            "preprocess_hooks": [
                
            ],
            "postprocess_hooks": [
            ]
        },
        "type": "constant_weights",
        "cls": "ConstantWeightsOutputHead",
        "args": {
            "num_classes": {
                "index": 2, "value": 2
            },
            "channels": {
                "index": 3, "value": [512, 768, 512]
            },
            "use_img_size": {
                "index": 4, "value": [False, True]
            }
        },
        "kwargs": {
            "load_from": None
        }
    },
    "trainer": {
        "criterion": {
            "set_cost_class": 2,
            "set_cost_bbox": 5,
            "set_cost_giou": 2,
            "num_classes": 2,
            "focal_alpha": 0.25
        },
        "batch_size": 1,
        "seed": 42,
        "num_workers": 1,
        
        "lr_backbone_names": ["backbone.0"],
        "lr_linear_proj_names": ['reference_points', 'sampling_offsets'],
        "lr": 2e-4,
        "lr_backbone": 2e-5,
        "lr_linear_proj_mult": 0.1,
        "weight_decay": 1e-4,
        "epochs": 100,
        "lr_drop": 40,
        "lr_drop_epochs": None,
        "clip_max_norm": 0.1,
        
        "sgd": True,

        "frozen_weights": None,
        
        "output_dir": 'checkpoints',
        "resume": '',
        "eval": True
    },
    "datasets": [
    {
        "dataset": {
            "type": "eval",
            "pck": "hod",
            "cls": "HODataset",
            "builder": "build", #static method for building dataset, if None then build normally
            "args": {
                "image_set": {"index": 0, "value": 'train'},
                "dataset_path": {"index": 1, "value": 'data'},
                "train_anns": {"index": 2, "value": 'hod_anns_coco_train.json'},
                "val_anns": {"index": 3, "value": 'hod_anns_coco_test.json'}
            },
            "kwargs": {
                
            }
        }
    },
    {
        "dataset": {
            "type": "train",
            "pck": "hod",
            "cls": "HODataset",
            "builder": "build", #static method for building dataset, if None then build normally
            "args": {
                "image_set": {"index": 0, "value": 'train'},
                "dataset_path": {"index": 1, "value": 'data'},
                "train_anns": {"index": 2, "value": 'hod_anns_coco_train.json'},
                "val_anns": {"index": 3, "value": 'hod_anns_coco_test.json'}
            },
            "kwargs": {
                
            }
        }
    }
    ]
}