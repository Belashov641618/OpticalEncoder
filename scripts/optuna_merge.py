import sys
import os
import optuna
import argparse


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.argv.append('--output')
        sys.argv.append('optuna_merge/DBmerged.db')
        sys.argv.append('--input')
        if os.path.exists('optuna_merge/DBmerged.db'): os.remove('optuna_merge/DBmerged.db')
        for filepath in os.listdir('optuna_merge'):
            if filepath.endswith('.db'):
                sys.argv.append(f"optuna_merge/{filepath}")


    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help="Путь к выходному файлу")
    parser.add_argument('--input', type=str, nargs='+', required=True, help="Пути к входным файлам")
    args = parser.parse_args()
    output_file = "sqlite:///" + os.path.abspath(os.path.expanduser(args.output))
    input_files = ["sqlite:///" + os.path.abspath(os.path.expanduser(file)) for file in args.input]
    print(f"Output file: {output_file}")
    print(f"Input files:\n\t" + "\n\t".join([f"{input_file} | {os.path.exists(input_file[len('sqlite:///'):])}" for input_file in input_files]))

    studies_names = set()
    for input_file in input_files:
        studies_names.update(set(optuna.get_all_study_names(input_file)))
    print(studies_names)

    if os.path.exists(output_file): os.remove(output_file)
    for study_name in studies_names:
        trials = []
        trials_params = set()
        trials_attributes = set()
        trials_values = set()
        for input_file in input_files:
            if study_name in optuna.get_all_study_names(input_file):
                for trial in optuna.load_study(study_name=study_name, storage=input_file).get_trials():
                    params_trigger = tuple(trial.params.items()) in trials_params
                    attributes_trigger = tuple(trial.user_attrs.items()) in trials_attributes
                    values_trigger = trial.value in trials_values
                    if not all([params_trigger, attributes_trigger, values_trigger]):
                        trials_params.update([tuple(trial.params.items())])
                        trials_attributes.update([tuple(trial.user_attrs.items())])
                        trials_values.update([trial.value])
                        trials.append(trial)
        study_template = optuna.load_study(study_name=study_name, storage=input_files[0])
        study = optuna.create_study(storage=output_file, study_name=study_name, direction=study_template.direction, load_if_exists=False)
        study.add_trials(trials)
    print("Done!")