import json
import argparse
import pandas as pd
import utils.common as common
from evals.mmlu_eval import MMLUEval
# from sampler.chat_completion_sampler import (
#     OPENAI_SYSTEM_MESSAGE_API,
#     OPENAI_SYSTEM_MESSAGE_CHATGPT,
#     ChatCompletionSampler,
# )
from sampler.qwen_chat_sampler import (
    QWEN_SYSTEM_MESSAGE,
    QwenApplyChatSampler
)
from utils._types import Mode

from datetime import datetime
from pytz import timezone

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--shuffle", action="store_true", help="Run in shuffle mode")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument("--mode", type=str, help="Select mode (full, half, option_only, question_only)")
    parser.add_argument(
        "--groups", 
        nargs="*",
        type=str,
        help="Select specific list of groups (optional)"
    )
    parser.add_argument(
        "--subjects", 
        nargs="*",
        type=str,
        help="Select specific list of subjects (optional)"
    )

    args = parser.parse_args()

    models = {
        # chatgpt models:
        # "gpt-4o-2024-11-20_assistant": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-2024-11-20_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        # ),
        # "gpt-4o_assistant": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-mini": ChatCompletionSampler(
        #     model="gpt-4o-mini-2024-07-18",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # qwen models
        "qwen": QwenApplyChatSampler(
            model="Qwen/Qwen2.5-7B-Instruct",
            system_message=QWEN_SYSTEM_MESSAGE,
            max_tokens=256,
        )
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    match args.mode:
        case "full":
            mode = Mode.FULL
        case "half":
            mode = Mode.HALF
        case "option_only":
            mode = Mode.OPTION_ONLY
        case "question_only":
            mode = Mode.QUESTION_ONLY
        case _:
            raise Exception(f"Unrecognized mode type: {args.mode}")

    def get_evals(eval_name, debug_mode):
        num_examples = 0  # default zero shot
        match eval_name:
            case "mmlu":
                return MMLUEval(mode, num_examples=0 if debug_mode else num_examples, 
                                groups=args.groups, subjects=args.subjects, is_shuffle=args.shuffle)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in ['mmlu']
    }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    shuffle_suffix = "_SHUFFLE" if args.shuffle else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            now = datetime.now(timezone('Asia/Jakarta'))
            eval_time = now.strftime("%d-%m-%Y_%H-%M-%S")
            result = eval_obj(sampler)
            # ^^^ how to use a sampler         
            file_stem = f"result/{mode.value}_{eval_name}_{model_name}"
            if args.subjects:
                file_stem += f"_{args.subjects}"
            elif args.groups:
                file_stem += f"_{args.groups}"
            file_stem += f"_{eval_time}"

            report_filename = f"{file_stem}{debug_suffix}{shuffle_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"{file_stem}{debug_suffix}{shuffle_suffix}.json"
            result.df_result.to_csv(f'{file_stem}{debug_suffix}{shuffle_suffix}.csv', index=False)
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("accuracy", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
