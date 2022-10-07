SYN_THRESHOLD = 30
RI_THRESHOlD = 50
GENES_DIM = 5001
EPOCHS = 500
HYPERPARAMETERS = [
    # {
    #     'learning_rate': 0.005,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'learning_rate': 0.001,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'learning_rate': 0.0005,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'learning_rate': 0.0001,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5lr
    # },
    {
        'lr': 0.0001,
        'batch_size': 256,
        'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
        'drop_out': 0.5
    },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    {
        'lr': 0.0001,
        'batch_size': 256,
        'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
        'drop_out': 0.5
    },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.0001,
    #     'batch_size': 512,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 512,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00001,
    #     'batch_size': 256,
    #     'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
    #     'drop_out': 0.5
    # }

]

MULTITASK_HYPERPARAMETERS=[
    # {
    #     'lr': 0.005,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.001,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.0005,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.0001,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [512, 512, 128, 256, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.0001,
    #     'batch_size': 256,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.0001,
    #     'batch_size': 512,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 512,
    #     'hidden_dims': [2048, 1024, 1024, 128, 512, 64],
    #     'drop_out': 0.5
    # },
    {
        'lr': 0.0001,
        'batch_size': 256,
        'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
        'drop_out': 0.5
    },
    # {
    #     'lr': 0.00005,
    #     'batch_size': 256,
    #     'hidden_dims': [4096, 2048, 2048, 128, 1024, 64],
    #     'drop_out': 0.5
    # }

]

SYNERGY_HYPERPARAMETERS=[
    {
        'lr': 0.0001,
        'batch_size': 256,
        'hidden_dims': [1024, 128],
        'drop_out': 0.5
    },
    {
        'lr': 0.0001,
        'batch_size': 256,
        'hidden_dims': [2048, 128],
        'drop_out': 0.5
    },
]
