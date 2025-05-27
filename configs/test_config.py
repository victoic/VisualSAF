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
                {
                    "package": "general_dl.detectors",
                    "module": "postprocess_hooks",
                    "hook": "separate_outputs",
                    "args": {
                    },
                    "kwargs": {
                    }
                }
            ]
        },
        "type": "deformable-detr",
        "cls": "DeformableDETR",
        "args": {
            "num_classes": {"index": 0, "value": 91},
            "num_queries": {"index": 1, "value": 300},
            "num_feature_levels": {"index": 2, "value": 4},
            "detr_args": {"index": 3, "value": {
                "num_classes": 91,
                "lr": 2e-4,
                "lr_backbone_names": ["backbone.0"],
                "lr_backbone": 2e-5,
                "lr_linear_proj_names": ["reference_points", "sampling_offsets"],
                "lr_linear_proj_mult": 0.1,
                "batch_size": 2,
                "weight_decay": 1e-4,
                "epochs": 50,
                "lr_drop": 40,
                "lr_drop_epochs": None,
                "clip_max_norm": 0.1,
                "sgd": True,
                "frozen_weights": None,
                "backbone": "resnet50",
                "dilation": True,
                "position_embedding": "sine",
                "position_embedding_scale":  6.283185307179586,
                "num_feature_levels": 4,
                "enc_layers": 6,
                "dec_layers": 6,
                "dim_feedforward": 1024,
                "hidden_dim": 256,
                "dropout": 0.1,
                "nheads": 8,
                "num_queries": 300,
                "dec_n_points": 4,
                "enc_n_points": 4,
                "masks": False,
                "set_cost_class": 2,
                "set_cost_bbox": 5,
                "set_cost_giou": 2,
                "mask_loss_coef": 1,
                "dice_loss_coef": 1,
                "cls_loss_coef": 2,
                "bbox_loss_coef": 5,
                "giou_loss_coef": 2,
                "focal_alpha": 0.25
                }
            }
        },
        "kwargs": {
            "load_from": "r50_deformable_detr-checkpoint.pth"
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
                    {
                        "package": "semantic_variables",
                        "module": "preprocess_hooks",
                        "hook": "tensor_from_nested",
                        "args": {
                        },
                        "kwargs": {
                        }
                    }
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
                    {
                        "package": "semantic_variables",
                        "module": "preprocess_hooks",
                        "hook": "tensor_from_nested",
                        "args": {
                        },
                        "kwargs": {
                        }
                    }
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
                "load_from": "best_model_nyu.ckpt"
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
        },
        "detector_outputs": [4, 91]
    },
    "trainer": {
        "criterion": {
            "set_cost_class": 2,
            "set_cost_bbox": 5,
            "set_cost_giou": 2,
            "num_classes": 2,
            "focal_alpha": 0.25,
            "mask_loss_coef": 1,
            "dice_loss_coef": 1,
            "cls_loss_coef": 2,
            "bbox_loss_coef": 5,
            "giou_loss_coef": 2
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
        "eval": False,
        "distributed": False,
        "start_epoch": 0
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