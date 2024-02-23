import time
import fire
import os
import glob

CONFIGS = {
    'arc': {
        'tasks': 'arc_challenge',
        'n_shots': 25,
        'metric_name': 'acc_norm'
    },
    'hellaswag': {
        'tasks': 'hellaswag',
        'n_shots': 10,
        'metric_name': 'acc_norm'
    },
    'truthfulqa': {
        'tasks': 'truthfulqa_mc',
        'n_shots': 0,
        'metric_name': 'mc2'
    },
    'mmlu': {
        'tasks': 'hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions',
        'n_shots': 5,
        'metric_name': 'acc'
    },
}
BATCH_SIZE = 32


def evaluate(config_name, model_dir, output_path):
    config = CONFIGS[config_name]
    tasks, n_shots, metric_name = \
        config['tasks'], config['n_shots'], config['metric_name']

    batch_size = BATCH_SIZE

    while True:
        command = f'python eval_utils/harness.py '\
                  f'--model=hf-causal '\
                  f'--model_args=\"pretrained={model_dir}\" '\
                  f'--tasks={tasks} '\
                  f'--num_fewshot={n_shots} '\
                  f'--batch_size={batch_size} '\
                  f'--output_path={output_path} '\
                  f'--no_cache'

        if os.system(command) == 0:
            break
        else:
            print(f'COMMAND \"{command}\" failed. rerunning...')
            if batch_size > 1:
                batch_size = batch_size // 2


def main(workdir='workdir_7b'):
    while True:
        ckpt_dirs = glob.glob(f'{workdir}/ckpt_*')
        ckpt_dirs.sort(key=lambda s: int(s[len(f'{workdir}/ckpt_'):]))

        for model_dir in ckpt_dirs:
            for config_name in CONFIGS.keys():
                output_path = f'{model_dir}/eval_{config_name}.json'
                if not os.path.exists(output_path):
                    print(f'evaluating {config_name} for {model_dir}...')
                    print('running...', file=open(output_path, 'w'), flush=True)

                    time.sleep(60)
                    evaluate(
                        config_name=config_name,
                        model_dir=model_dir,
                        output_path=output_path)


if __name__ == '__main__':
    fire.Fire(main)