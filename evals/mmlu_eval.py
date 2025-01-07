"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

# import blobfile as bf
import pandas

import utils.common as common
from utils.common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from utils._types import Eval, EvalResult, SamplerBase, SingleEvalResult, Mode

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

category2subject = {
    'stem': ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 
             'college_computer_science', 'college_mathematics', 'college_physics', 
             'computer_security', 'conceptual_physics', 'electrical_engineering', 
             'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 
             'high_school_computer_science', 'high_school_mathematics', 
             'high_school_physics', 'high_school_statistics', 'machine_learning'], 
    'other': ['anatomy', 'business_ethics', 'clinical_knowledge', 
              'college_medicine', 'global_facts', 'human_aging', 
              'management', 'marketing', 'medical_genetics', 
              'miscellaneous', 'nutrition', 'professional_accounting', 
              'professional_medicine', 'virology'], 
    'social_sciences': ['econometrics', 'high_school_geography', 
                        'high_school_government_and_politics', 
                        'high_school_macroeconomics', 
                        'high_school_microeconomics', 
                        'high_school_psychology', 'human_sexuality', 
                        'professional_psychology', 'public_relations', 
                        'security_studies', 'sociology', 'us_foreign_policy'], 
    'humanities': ['formal_logic', 'high_school_european_history', 
                   'high_school_us_history', 'high_school_world_history', 
                   'international_law', 'jurisprudence', 'logical_fallacies', 
                   'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 
                   'professional_law', 'world_religions']
}


class MMLUEval(Eval):
    def __init__(self, mode: Mode, num_examples: int | None = None, language: str = "EN-US",
                 groups: str = None, subjects: str = None, is_shuffle=False):
        # if language != "EN-US":
        #     url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        # else:
        #     url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        # df = pandas.read_csv(bf.BlobFile(url))
        self.mode = mode
        df_path = ""
        if mode == Mode.HALF:
            df_path = 'training-eval/data/mmlu_half'
        else:
            df_path = 'training-eval/data/mmlu_full'
        if is_shuffle:
            df_path += '_shuffled'
        
        df = pandas.read_csv(f"{df_path}.csv")
        if subjects:
            self.df = df[df['Subject'].isin(subjects)] 
        elif groups:
            self.df = df[df['Group'].isin(groups)]     
        else:
            self.df = df

        self.df['Extracted Answer'] = 'invalid'
        self.df['Score'] = 0.0
        examples = [row.to_dict() for _, row in self.df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row, self.mode), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages))
            extracted_answer = None
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                match = re.search(regex, response_text)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            if extracted_answer:
                self.df.loc[self.df.id == row['id'], 'Extracted Answer'] = extracted_answer
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            self.df.loc[self.df.id == row['id'], 'Score'] = score
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                group=row["Group"],
                subject=row["Subject"],
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results, self.df)
