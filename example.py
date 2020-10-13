import glob
import os

from gin_factory import GinFactory

"""
In this example we will create gin files for multiple train/validation pairs. We will start numbering them from 000.
Every second file will be a train file, and every second - validation.
"""

output_naming_scheme = 'numerical'  # default, no support for different schemes, but should be easy to add
numerical_scheme_digits = 3  # default, will make name files contain 3 digits, e.g. 000.gin, 001.gin, 123.gin
numerical_scheme_every = 2  # 1 is the default, but we will first make a gin file for each training config, then use them to create val files with number +=1

gin_base_file = 'base_configs/base_config.gin'
output_folder = './gin_created_files/'

"""Training files"""
# Args used just for writing/overwriting the gin file
train_stable_args = {
    'x2.a1': ['benmal_auc_average', 'benmal_auc_malignant'],
    'x2.a2': "Train",
}
# Args used to perform a param search
train_varying_args = {
    'x3.a1': [1e-2, 1e-3, 1e-4],
    'x3.a2': [[('ssim_loss', 0.9), ('l1_loss', 0.095), ('nll_2label_loss', 0.005)],
              [('ssim_loss', 0.9), ('l1_loss', 0.085), ('nll_2label_loss', 0.015)]],
}

factory = GinFactory(output_naming_scheme, numerical_scheme_digits, numerical_scheme_every)
factory.create_multiple_gin_files(output_folder,
                                  gin_base_file,
                                  train_stable_args,
                                  train_varying_args,
                                  first_file_number=0)

"""
By now, we have created 3x2=6 gin files named 000.gin, 002.gin, ..., 008.gin, 010.gin.
Now, let's use our factory, to create validation files for each created gin file: 001.gin, 003.gin, ..., 009.gin, 011.gin.
"""

for train_file in glob.glob(os.path.join(output_folder, '*')):
    train_filename = os.path.basename(train_file)
    train_number = train_filename[:-4]

    val_stable_args = {
        'x2.a1': ['val_benmal_auc_malignant', 'val_benmal_prauc_malignant'],
        'x2.a2': "Valid",
        'x2.a3': f'./experiments/{train_number}/model_best_val.pt',
        'x3.a1': 0,
    }
    val_file_number = int(train_filename[:-4]) + 1
    factory.create_multiple_gin_files(output_folder, gin_base_file, val_stable_args, None, val_file_number)

"""
Be careful! Using this code twice will assume each previously created file is a train file. Keep that in mind if you update the code.
"""
