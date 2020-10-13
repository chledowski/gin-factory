import itertools
import os
from typing import List, Dict, Any, Optional


class GinFactory:
    """Class used to create multiple gin files automatically."""

    def __init__(self,
                 output_naming_scheme: str = 'numerical',
                 numerical_scheme_digits: Optional[int] = 3,
                 numerical_scheme_every: Optional[int] = 1,
    ):
        """Initialize class.

        :param output_naming_scheme: Naming scheme for the output files.
        :param numerical_scheme_digits: Maximum number of digits used to create gin files in numerical scheme.
        """
        self.output_naming_scheme = self.get_scheme(output_naming_scheme,
                                                    numerical_scheme_digits)
        self.numerical_scheme_digits = numerical_scheme_digits
        self.numerical_scheme_every = numerical_scheme_every

    def __call__(self, *args, **kwargs):
        self.create_multiple_gin_files(*args, **kwargs)

    def create_multiple_gin_files(self,
                                  output_folder: str,
                                  base_file: Optional[str] = None,
                                  stable_args: Optional[Dict[str, Any]] = None,
                                  varying_args: Optional[Dict[str, List[Any]]] = None,
                                  first_file_number: Optional[int] = 0):
        """Main function that takes base_file, stable_args and varying_args and uses them to produce new gin files.

        :param stable_args: arguments that will not be varying.
        :param varying_args: args with at least two values.
        :param output_folder: path to the final destination of the gin files.
        :param base_file: path to the gin file, that will be used as a base.
         This means, if we will not overwrite the values from this file, they will be in the final file.
        :param first_file_number: number of the first file (used with numerical scheme).
        """
        if stable_args and varying_args:
            intersection = set(stable_args).intersection(set(varying_args))
            assert len(intersection) == 0, f'stable_args and varying_args cannot have matching keys (found: {intersection}).'

        final_args_dict = {}
        # Loading args from base_file
        if base_file:
            with open(base_file, 'r') as f:
                for line in f:
                    processed_line = self.process_gin_line(line)
                    if processed_line:
                        final_args_dict[processed_line[0]] = processed_line[1]

        # Loading args from stable_args
        if stable_args:
            for name in stable_args:
                final_args_dict[name] = stable_args[name]

        # Loading args from varying_args
        if varying_args:
            varying_keys = list(varying_args.keys())
            varying_vals = [varying_args[key] for key in varying_args]
            combinations = list(itertools.product(*varying_vals))

            assert len(combinations) + first_file_number < 10 ** self.numerical_scheme_digits, \
                f"Number of the last file ({len(combinations) + first_file_number}) would have more digits than the maximum " \
                f"amount {self.numerical_scheme_digits}. Please increase the numerical_scheme_digits."

            file_number = first_file_number
            for combination in combinations:
                for i in range(len(varying_keys)):
                    final_args_dict[varying_keys[i]] = combination[i]
                self.save_args(final_args_dict, output_folder, file_number)
                file_number += self.numerical_scheme_every
        else:
            self.save_args(final_args_dict, output_folder, first_file_number)

    def save_args(self, args_dict, output_folder, file_number, sort_method=None):
        """Save the arguments to a file.

        :param args_dict: arguments to save.
        :param output_folder: output folder.
        :param file_number: number of file (used in numerical scheme).
        :param sort_method: method to sort the keys in the output gin file. Not supported yet.
        :return:
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_filename = os.path.join(output_folder, self.output_naming_scheme(file_number))

        with open(save_filename, 'w') as f_save:
            for key in args_dict:
                output = f'{key}={args_dict[key]}\n'
                f_save.write(output)

    @staticmethod
    def process_gin_line(line):
        """Delete comments, strip from '\n' and convert empty lines to None.
        :param line: Input line.
        """
        line = line.strip('\n').split('#')[0]
        if '=' in line:
            name, value = line.split('=', maxsplit=1)
            return name, value
        else:
            return None

    @staticmethod
    def get_scheme(name, numerical_scheme_digits):
        """This function creates a function for gin file naming.

        :param name:
        :param numerical_scheme_digits:
        :return:
        """
        if name == 'numerical':
            def numerical_scheme(k):
                num_digits = len(str(k))
                if num_digits <= numerical_scheme_digits:
                    extra_zeros = numerical_scheme_digits - num_digits
                    return extra_zeros * '0' + str(k) + '.gin'
                else:
                    raise ValueError(f"Number {k} has more digits than the numerical_scheme_digits argument ({numerical_scheme_digits}). Please fix this.")

            return numerical_scheme


if __name__ == '__main__':
    output_naming_scheme = 'numerical'
    numerical_scheme_digits = 3
    numerical_scheme_every = 1
    gin_base_file = './configs/dbt_3d_to_2d_cnvclf_cnv=3dcnn_clf=resnetv2_ensemble_loss.gin'
    output_folder = './gin_created_files/'
    stable_args = {
        'key1': 'val1',
        'key2': True,
        'key3': [2, 4, 6],
        'key4': [],
        'train.loss': [('ssim_loss', 0.99), ('l1_loss', 0.01), ('nll_2label_loss', 0)],
    }
    varying_args = {
        'vkey1': ['val11', 'val12'],
        'vkey2': ['val21', 'val22'],
        'vkey3': ['val31', 'val32'],
        'vkey4': ['val41', 'val42'],
    }

    factory = GinFactory(output_naming_scheme, numerical_scheme_digits, numerical_scheme_every)
    factory.create_multiple_gin_files(output_folder, gin_base_file, stable_args, varying_args, 0)
