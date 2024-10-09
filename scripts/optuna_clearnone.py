import argparse
import os.path
import sys
import json
import optuna
import numpy

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.argv.append('--output')
        sys.argv.append('optuna_clear/DBcleared.db')
        sys.argv.append('--input')
        sys.argv.append('optuna_clear/DB.db')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Путь к входному файлу")
    parser.add_argument('--output', type=str, required=True, help="Путь к выходному файлу")
    args = parser.parse_args()
    input_file = "sqlite:///" + args.input
    output_file = "sqlite:///" + args.output

    if os.path.exists(output_file): os.remove(output_file)

    studies_names = optuna.get_all_study_names(input_file)
    for study_name in studies_names:
        trials = []
        study = optuna.load_study(study_name=study_name, storage=input_file)
        i = 0
        for trial in study.trials:
            state = trial.state
            value = trial.value
            intermediate = trial.intermediate_values
            number = trial.number
            complete = trial.datetime_complete
            start = trial.datetime_start
            params = trial.params
            attrs = trial.user_attrs
            system = trial.system_attrs
            distributions = trial.distributions

            if state != optuna.trial.TrialState.COMPLETE: continue
            if value >= 100.0:
                print("Found trial with incorrect value")
                print(f"\tBest accuracy: {value}")
                print(f"\tEpochs count: {params['epochs']}")
                accuracies_history = json.loads(attrs['accuracies_history'])
                print("\tHistory: " + ", ".join([f'{round(accuracy,2)}' for accuracy in accuracies_history]))
                confusion_matrices = numpy.array(json.loads(attrs['confusion_matrixes']))
                confusion_matrices_sums = numpy.sum(confusion_matrices, axis=(1,2))
                print("\tConfusion matrices sums: " + ", ".join([f"{round(cm_sum, 2)}" for cm_sum in confusion_matrices_sums]))
                confusion_matrices_sums_set = list(set(confusion_matrices_sums))
                confusion_matrices_sums_count = [0 for _ in range(len(confusion_matrices_sums_set))]
                for confusion_matrix_sum in confusion_matrices_sums:
                    index = confusion_matrices_sums_set.index(confusion_matrix_sum)
                    confusion_matrices_sums_count[index] += 1
                confusion_matrix_correct_sum = confusion_matrices_sums_set[confusion_matrices_sums_count.index(max(confusion_matrices_sums_count))]
                confusion_matrices_incorrect = [i for i, cm_sum in enumerate(confusion_matrices_sums) if cm_sum != confusion_matrix_correct_sum]
                print("\tConfusion matrices sums rates:" + ", ".join([f"({round(cm_sum,2)} : {int(sm_rate)})" for cm_sum, sm_rate in zip(confusion_matrices_sums_set, confusion_matrices_sums_count)]))
                print(f"\tLet {round(confusion_matrix_correct_sum, 2)} be correct sum, so " + ", ".join([str(index) for index in confusion_matrices_incorrect]) + " incorrect")
                drop_from = min(confusion_matrices_incorrect)
                print(f"\tFirst incorrect is {drop_from}, dropping them:")
                loss_history = json.loads(attrs['loss_history'])
                loss_history_length = len(loss_history)
                loss_history_epoch_length = loss_history_length//params['epochs']
                params['epochs'] = params['epochs'] - len(confusion_matrices) + drop_from
                accuracies_history = accuracies_history[:drop_from]
                value = max(accuracies_history)
                attrs['accuracies_history'] = json.dumps(accuracies_history)
                confusion_matrices_amount = len(confusion_matrices)
                confusion_matrices = confusion_matrices[:drop_from]
                attrs['confusion_matrixes'] = json.dumps(confusion_matrices.tolist())
                attrs['loss_history'] = json.dumps(loss_history[:loss_history_epoch_length*params['epochs']])
                print(f"\t\tBest accuracy set to {value}")
                print(f"\t\tEpochs count set to {params['epochs']}")
                print(f"\t\tAccuracies history set to " + ", ".join([f'{round(accuracy,2)}' for accuracy in accuracies_history]))
                print(f"\t\tConfusion matrices modified: {confusion_matrices_amount} -> {len(confusion_matrices)}")
                print(f"\t\tLoss history cropped: {loss_history_length} -> {len(loss_history[:loss_history_epoch_length*params['epochs']])}")

            trials.append(optuna.trial.FrozenTrial(
                number=number,
                state=state,
                value=value,
                datetime_start=start,
                datetime_complete=complete,
                params=params,
                distributions=distributions,
                user_attrs=attrs,
                system_attrs=system,
                intermediate_values=intermediate,
                trial_id=i,
            ))

            i += 1
        study_template = optuna.load_study(study_name=study_name, storage=input_file)
        study = optuna.create_study(storage=output_file, study_name=study_name, direction=study_template.direction, load_if_exists=False)
        study.add_trials(trials)
    print("Done!")
