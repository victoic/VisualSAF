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
                {
                    "package": "general_dl.backbone",
                    "module": "preprocess_hooks",
                    "hook": "resnet_transform",
                    "args": {
                        
                    },
                    "kwargs": {

                    }
                }
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
                    "package": "general_dl.detector",
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
            "num_classes": {"index": 0, "value": 80},
            "num_queries": {"index": 1, "value": 300},
            "num_feature_levels": {"index": 2, "value": 4},
            "detr_args": {"index": 3, "value": {
                "num_classes": 80,
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
            }}
        },
        "kwargs": {
            "aux_loss": False,
            "with_box_refine": False,
            "two_stage": False
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
                        "package": "semantic_variables.scene_recognition",
                        "module": "preprocess_hooks",
                        "hook": "densenet_transform",
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
                        "package": "semantic_variables.monocular_depth_estimation",
                        "module": "preprocess_hooks",
                        "hook": "glpdepth_transform",
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
    }
}